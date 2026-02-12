from collections.abc import Callable
from typing import Any
import requests
import logging
import re
import os
from dotenv import load_dotenv
from .model import Cell, Entry, Candidate

# Module-level hook variable (not class-level) so it can be accessed by staticmethods
_generate_candidates_hook: Callable[[Entry, int, int], list[Any]] | None = None

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("src.llm")

def normalize_candidate(answer: str) -> str:
    """Normalize a candidate answer: capitalize and remove spaces, punctuation, and digits.
    
    Returns only alphabetic characters in uppercase.
    """
    return ''.join(ch.upper() for ch in answer if ch.isalpha())

class LLM:
    # Configuration for candidate generation behavior
    MAX_WIDENING: int = 0
    MAX_CANDIDATES: int = 5
    # Load environment variables from .env
    load_dotenv()
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama3.1:8b")
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    #logger.debug("Using Ollama model " + MODEL_NAME)

    # Optional hook for overriding candidate generation (used by simulated/test puzzles).
    # When set, generate_candidates() dispatches to this hook instead of calling the LLM.
    # This enables deterministic, repeatable puzzle solves for testing and demos.
    _generate_candidates_hook: Callable[[Entry, int, int], list[Any]] | None = None

    @staticmethod
    def set_generate_candidates_hook(
            hook: Callable[[Entry, int, int], list[Any]] | None
    ) -> None:
        """
        Install or clear a custom candidate generation hook.
        
        When a hook is installed, generate_candidates() will call the hook instead
        of the real LLM logic. This is primarily used for:
        - Simulated puzzles with pre-defined candidate lists
        - Test fixtures that need deterministic behavior
        - UI demos that don't require an actual LLM connection
        
        Pass None to clear the hook and restore default (stub) behavior.
        """
        global _generate_candidates_hook
        _generate_candidates_hook = hook


    @staticmethod
    def generate_candidates(
            entry: Entry,
            widening_level: int,
            max_candidates: int | None = None,
    ) -> list[Candidate]:
        """
        Given all the information we have about a particular crossword clue, prompt the LLM for a
            list of potential answers ("candidates").
        
        This uses a two-step approach:
        1. Generate candidate answers (generation phase)
        2. Score each candidate independently (evaluation phase)
        
        This separation improves calibration by having the LLM evaluate candidates in a
        distinct context, where it tends to be more conservative and accurate.
        
        :param entry: The crossword clue for which candidate answers are being generated.   The Entry
            object contains lots of contexual info we pass to the LLM in the prompt, including the 
            answer's length, the pattern of known crossing letters, the list of "hints" obtained 
            from the vector DB, and of course the clue itself.
        :param widening_level: Measures how "creative" we want the LLM to be.  The first time we ask,
            it's at the minimum level (0), but if the LLM is unable to produce any viable candidates,
            we increase ("widen") this value.  The effect is that at higher levels, the prompt we pass
            to the LLM will have additional instructions, like "consider multiword answers".  The
            widenening level is unique for each different pattern of letters passed to the LLM.
        :param max_candidates: The maximum number of candidate answers allowed.
        :return: A list of candidate answers, paired with independently-evaluated confidence ratings.
        """
        if max_candidates is None:
            max_candidates = LLM.MAX_CANDIDATES
        if _generate_candidates_hook is not None:
            return _generate_candidates_hook(entry, widening_level, max_candidates)

        # Step 1: Generate candidate answers (without embedded scoring)
        answers = LLM._generate_candidate_answers(entry, widening_level, max_candidates)
        if not answers:
            return []
        
        # Step 2: Score each candidate independently
        scored_candidates = LLM._score_candidates(entry, answers)
        
        # Debug print: show all candidates and their confidence levels
        logger.debug("[LLM RESPONSE]: " + ", ".join(f"{c.answer} ({c.confidence})" for c in scored_candidates))
        return scored_candidates

    @staticmethod
    def _generate_candidate_answers(
            entry: Entry,
            widening_level: int,
            max_candidates: int,
    ) -> list[str]:
        """
        Generate candidate answers for a clue (generation phase only, no scoring).
        
        :param entry: The crossword entry
        :param widening_level: How creative to be with candidates
        :param max_candidates: Maximum number of candidates to generate
        :return: List of normalized candidate answers (strings)
        """
        prompt: str = LLM.create_prompt(entry, widening_level)
        
        def matches_pattern(candidate: str, pattern: str) -> bool:
            return all(p == "." or p == c for c, p in zip(candidate, pattern))

        try:
            logger.debug(f"[LLM GENERATE] Clue: '{entry.clue}' | Length: {entry.length} | Pattern: {entry.pattern}")
            if entry.hints:
                hints_str = ", ".join(f"'{clue}'->{answer}" for clue, answer in entry.hints)
                logger.debug(f"[LLM GENERATE HINTS] {hints_str}")
            #logger.debug(f"[LLM GENERATE PROMPT]\n{prompt}")
            response = requests.post(
                LLM.OLLAMA_URL,
                json = {
                    "model": LLM.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "")
            #logger.debug(f"[LLM GENERATE RAW RESPONSE]\n{output}")

            # Parse answers (no confidence in generation phase)
            answers: list[str] = []
            for part in output.split("\n"):
                part = part.strip()
                if not part:
                    continue
                # Remove pipe and everything after it if present (in case LLM includes it anyway)
                if "|" in part:
                    part = part.split("|", 1)[0].strip()
                answer = normalize_candidate(part)
                # Only include answers of the correct length
                if len(answer) == entry.length and answer not in answers:
                    answers.append(answer)
                if len(answers) >= max_candidates:
                    break
            hint_matches: list[str] = []
            if entry.hints:
                for _, hint_answer in entry.hints:
                    normalized_hint = normalize_candidate(hint_answer)
                    if len(normalized_hint) != entry.length:
                        continue
                    if not matches_pattern(normalized_hint, entry.pattern):
                        continue
                    if normalized_hint not in hint_matches:
                        hint_matches.append(normalized_hint)

            ordered = hint_matches + [answer for answer in answers if answer not in hint_matches]
            limit = max(max_candidates, len(hint_matches))
            ordered = ordered[:limit]
            logger.debug(f"[LLM GENERATE RESULT] Generated {len(ordered)} candidates: {ordered}")
            return ordered
        except Exception as e:
            logger.error(f"Ollama generation query failed: {e}")
            return []

    @staticmethod
    def _score_candidates(
            entry: Entry,
            answers: list[str],
    ) -> list[Candidate]:
        """
        Score a list of candidate answers independently (evaluation phase).
        
        Each candidate is evaluated separately for how well it fits the clue and pattern,
        which improves calibration compared to scoring during generation.
        
        :param entry: The crossword entry
        :param answers: List of candidate answers to score
        :return: List of Candidate with confidence ratings
        """
        if not answers:
            return []
        
        clue = entry.clue
        hints = entry.hints
        
        # Build a prompt for evaluating these specific candidates
        prompt = "TASK: Evaluate how well each CANDIDATE answer fits the CLUE and PATTERN.\n"
        prompt += "\nRULES:\n"
        prompt += "- For each CANDIDATE, determine if it is a plausible answer for the CLUE.\n"
        prompt += "- HINTS are provided for context but are not exhaustive.\n"
        prompt += "- Be conservative: rate candidates lower if there is any doubt.\n"
        prompt += "\nCONFIDENCE RUBRIC:\n"
        prompt += "- 90-100: Definitive answer; very confident match.\n"
        prompt += "- 80-89: Strong match with subtle interpretation.\n"
        prompt += "- 50-79: Plausible but less certain.\n"
        prompt += "- Below 50: Speculative or uncertain.\n"
        
        if hints:
            prompt += "\nHINTS (for context):\n"
            for hint_clue, hint_answer in hints:
                prompt += f"- '{hint_clue}' -> '{hint_answer}'\n"
        
        prompt += f"\nCLUE: {clue}\n"
        prompt += "\nCANDIDATES TO EVALUATE:\n"
        for answer in answers:
            prompt += f"- {answer}\n"
        
        prompt += "\nRESPONSE FORMAT:\n"
        prompt += "- For each candidate, provide: ANSWER | CONFIDENCE\n"
        prompt += "- ONLY provide the lines with answers and confidence, no other text.\n"
        
        try:
            logger.debug(f"[LLM SCORE] Scoring {len(answers)} candidates for clue '{clue}'")
            response = requests.post(
                LLM.OLLAMA_URL,
                json = {
                    "model": LLM.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "")
            
            # Parse scored candidates
            scored: dict[str, float] = {}
            for part in output.split("\n"):
                part = part.strip()
                if "|" in part:
                    answer, conf = part.split("|", 1)
                    answer = normalize_candidate(answer)
                    try:
                        confidence = int(conf.strip())
                        confidence = max(0, min(confidence, 100))
                    except ValueError:
                        confidence = 50
                    if answer in answers:
                        scored[answer] = float(confidence)
            
            # Return scored candidates, maintaining original order
            result_candidates: list[Candidate] = []
            for answer in answers:
                confidence = scored.get(answer, 25.0)  # Default to low confidence if not scored
                result_candidates.append(Candidate(answer=answer, confidence=confidence))
            
            logger.debug(f"[LLM SCORE RESULT] Scored candidates: {[(c.answer, c.confidence) for c in result_candidates]}")
            return result_candidates
        except Exception as e:
            logger.error(f"Ollama scoring query failed: {e}")
            # Fallback: return unscored candidates with default confidence
            return [Candidate(answer=answer, confidence=25.0) for answer in answers]

    @staticmethod
    def verify_answer(
            entry: Entry,
            answer: str
    ) -> bool:
        """
        Ask the LLM if the given answer is plausible for the given the clue.  We use this to verify
        answers that are completed when its last letter is filled in by another crossing clue
        (as opposed to being answered directly).  For example, if the last letter of 1 Down is
        filled in when we answer 1 Across, we want to make sure that the resulting word makes sense
        for 1 Down's clue.
        """
        clue = entry.clue
        length = entry.length
        
        prompt: str = "Given a crossword clue, expected answer length, and a candidate answer, decide if the answer is plausible.\n"
        prompt += (f"CLUE: {clue}\n")
        prompt += (f"EXPECTED LENGTH: {length}\n")
        prompt += (f"ANSWER: {answer}\n")
        prompt += "Respond ONLY with the word Yes or No and no other text.\n"

        try:
            logger.debug(f"[LLM VERIFY] Clue: '{clue}' | Answer: '{answer}' | Length: {length}")
            response = requests.post(
                LLM.OLLAMA_URL,
                json = {
                    "model": LLM.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30  # Increased timeout to 30 seconds
            )
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "")
            is_valid = output == "Yes"
            logger.debug(f"[LLM VERIFY] Result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return False

    @staticmethod
    def create_prompt(entry: Entry, widening_level: int) -> str:
        """
        Fashion an appropriate prompt for the LLM to deduce a list of candidate answers.
        We will provide explicit instructions, the clue, the constraints (e.g., the length of the 
        answer, any known letters, etc.), and the set of example clues and answers we got from the 
        vector database.  It's important that we tell the LLM that those clues and answers are
        just contextual hints, and not an exhaustive list of possible answers.

        A good example of this is the clue "In on, as a trend", which has as its correct 
        answer the admittedly awkward two-word answer "hip to".  The query of the vector database 
        produced several close-ish answers, but nothing exact, and all of them were just one word:
        "hip" was in there twice and "hep" and "hipper" once each.  We have to make sure the LLM
        uses these answers as a guide and doesn't simply try to pick one of them.  With a little
        luck it will be smart enough to deduce the correct answer.
        
        :param entry: The crossword clue for which candidate answers are being generated.  See
            generate_candidates() for a fuller description.
        :param widening_level: Measures how "creative" we want the LLM to be with its answers.  See
            generate_candidates() for a fuller description.
            NOTE: For now, the max_widening_level is 0 and this parameter is ignored.  We'll
            elaborate on more creative prompt generation later.
        :return: The prompt we will pass to the LLM.
        """
        clue: str = entry.clue
        length: int = entry.length
        pattern: str = entry.pattern
        hints: list[tuple[str,str]] | None = entry.hints

        matcher: str = f'[^{re.escape(".")}]'
        valid_pattern: bool = (re.search(matcher, entry.pattern) is not None)

        prompt =  "TASK: Given a crossword clue and contextual hints, deduce CANDIDATE crossword answers.\n"
        prompt += "\nRULES:\n"
        prompt += "- A CANDIDATE is a potential answer deduced for the TARGET CLUE.\n"
        prompt += "- Many correct crossword answers are multi-word phrases.\n"
        prompt += "- Actively consider multi-word answers when deducing CANDIDATES.\n"
        prompt += "- If the clue is plural (ends in 's', 'es', or clearly refers to multiple items), the CANDIDATE must be plural.\n"
        prompt += "- If the clue is singular, the CANDIDATE must be singular.\n"
        prompt += "- Do not return answers with mismatched plurality.\n"
        prompt += "- Normalize each CANDIDATE by removing all spaces and punctuation and converting to upper case.\n"
        prompt += (f"- A normalized CANDIDATE must be {length} characters.\n")
        if valid_pattern:
            prompt += (f"- A normalized CANDIDATE should match this PATTERN, where a period . is an unknown character: {pattern}\n")
            prompt += "- When a PATTERN has only one or two unknown letters, focus on finding CANDIDATES that match the PATTERN exactly.\n"
        prompt += "- HINTS are past crossword clue-answer pairs semantically similar to the TARGET CLUE.\n"
        prompt += "- HINTS are unranked, and may be only loosely related to the TARGET CLUE.\n"
        prompt += "- HINTS do not provide an exhastive list of CANDIDATES, but they should be given additional weight.\n"
        prompt += "- If any HINT answers match the LENGTH and PATTERN, you MUST include them in the CANDIDATES.\n"
        prompt += "- Order CANDIDATES from most likely to least likely.\n"
        #prompt += "- CANDIDATES may be inferred from general crossword knowledge and common idiomatic usage, even if not present in the HINTS.\n"
        #prompt += "- HINTS should be used to infer patterns or meanings.\n"
        #prompt += "- Generate creative and diverse CANDIDATES, even if unusual or speculative.\n"
        prompt += "OUTPUT FORMAT:\n"
        prompt += "- Provide a list of up to ten CANDIDATES.\n"
        prompt += "- Each CANDIDATE must be on its own line.\n"
        prompt += "- IMPORTANT: DO NOT provide any other text or confidence scores.\n"
        if hints is not None and len(hints) > 0:
            prompt += "\nHINTS:\n"
            for hint_clue, hint_answer in hints:
                prompt += f"- REFERENCE CLUE: '{hint_clue}', REFERENCE ANSWER: '{hint_answer}'\n"
        if valid_pattern:
            prompt += (f"\nPATTERN: {pattern}\n")
        prompt += "\nTARGET CLUE: " + clue + "\n"
        prompt += "\nLENGTH: " + str(length) + "\n"

        return prompt


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Manual test: print candidates from a live LLM call.
    clue = "Old-fashioned butter maker"
    answer = "CHURN"
    length = len(answer)
    grid: list[list[Cell]] = [[Cell(0, i) for i in range(length)]]
    entry = Entry("1A", clue, answer, grid, (0, 0), length)

    print(f"Clue: {entry.clue}")
    candidates = LLM.generate_candidates(entry, widening_level=0)
    print("Candidates:")
    for cand in candidates:
        print(f"- {cand.answer} ({cand.confidence})")
