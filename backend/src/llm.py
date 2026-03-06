from collections.abc import Callable
from typing import Any
import requests
import logging
import re
import os
import math
from dotenv import load_dotenv

from model import Cell, Entry, Candidate

# Module-level hook variable (not class-level) so it can be accessed by staticmethods
_generate_candidates_hook: Callable[[Entry, str, int], list[Candidate]] | None = None

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("src.llm")

def normalize_candidate(answer: str) -> str:
    """Normalize a candidate answer: capitalize and remove spaces, punctuation, and digits.
    
    Returns only alphabetic characters in uppercase.
    """
    return ''.join(ch.upper() for ch in answer if ch.isalpha())

class LLM:
    # Configuration for candidate generation behavior
    DEFAULT_CONFIDENCE = 30.0
    MAX_SEARCH_LEVEL: int = 2
    MAX_CANDIDATES: list[int] = [7, 12, 15]
    CANDIDATE_GENERATION_TUNING_PARAMS: list[dict[str,float|int]] = [
        {"temperature": 0.25, "top_p": 0.8, "top_k": 10},
        {"temperature": 0.70, "top_p": 0.9, "top_k": 30},
        {"temperature": 1.00, "top_p": 0.95, "top_k": 60}
    ]
    OLLAMA_TIMEOUT_SECONDS = 120
    OLLAMA_KEEP_ALIVE = "30m"
    TOP_LOGPROBS = 5

    # Load environment variables from .env
    load_dotenv()
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama3.1:8b")
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    #logger.debug("Using Ollama model " + MODEL_NAME)

    # Optional hook for overriding candidate generation (used by simulated/test puzzles).
    # When set, generate_candidates() dispatches to this hook instead of calling the LLM.
    # This enables deterministic, repeatable puzzle solves for testing and demos.
    _generate_candidates_hook: Callable[[Entry, str, int], list[Candidate]] | None = None


    @staticmethod
    def set_generate_candidates_hook(
            hook: Callable[[Entry, str, int], list[Candidate]] | None
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
            pattern: str,
            search_level: int
    ) -> list[Candidate]:
        """
        Given all the information we have about a particular crossword clue, prompt the LLM for a
            list of potential answers ("candidates").
        
        This uses a two-step approach:
        1. Generate candidate answers (generation phase)
        2. Score each candidate independently (evaluation phase)
        
        This separation improves calibration by having the LLM evaluate candidates in a
        distinct context, where it tends to be more conservative and accurate.
        
        :param entry: The crossword clue for which candidate answers are being generated.
            The Entry object contains lots of contexual info we pass to the LLM in the prompt,
            including the answer's length, the pattern of known crossing letters, the list of
            "hints" obtained from the vector DB, and of course the clue itself.
        :param search_level: Measures how "creative" we want the LLM to be.  The first time 
            we ask, it's at the minimum level (0), but if the LLM is unable to produce any
            viable candidates, we increase this value.  At higher levels, the API's tuning
            parameters (tempeature, top_p, top_k) are adjusted, and also the prompt we pass 
            to the LLM will have additional instructions, like "consider multiword answers".
            The search level is unique for each different pattern of letters passed to the LLM.
        :return: A list of candidate answers.
        """
        if _generate_candidates_hook is not None:
            return _generate_candidates_hook(entry, pattern, search_level)

        # LLM CALL #1: Generate candidate answers
        candidates = LLM._generate_candidate_answers(entry, pattern, search_level)
        if not candidates:
            return []
        
        # LLM CALL #2: Score each candidate independently
        scored_candidates = LLM._score_candidates(entry, candidates, search_level)

        # Debug print: show all candidates and their aggregate confidence levels
        logger.debug("[LLM FINAL RESPONSE]: " + ", ".join(f"{c.answer} ({c.confidence:.1f})" for c in scored_candidates))
        return scored_candidates


    @staticmethod
    def _generate_candidate_answers(
            entry: Entry,
            pattern: str,
            search_level: int
    ) -> dict[str, Candidate]:
        """
        Generate candidate answers for a clue (generation phase only, no scoring).
        
        :param entry: The crossword entry
        :param search_level: How creative to be with candidates
        :param max_candidates: Maximum number of candidates to generate
        :return: List of normalized candidate answers (strings)
        """
        prompt: str = LLM._create_prompt(entry, pattern, search_level)
        
        try:
            logger.debug(f"[LLM GENERATE] Entry: {entry.entry_id} | Clue: '{entry.clue}' | Length: {entry.length} | Pattern: {entry.pattern} | Search level {search_level}")
            if entry.hints:
                hints_str = ", ".join(f"'{clue}'->{answer}" for clue, answer in entry.hints)
                logger.debug(f"[LLM GENERATE HINTS] {hints_str}")
            #logger.debug(f"[LLM GENERATE PROMPT]\n{prompt}")
            response = requests.post(
                LLM.OLLAMA_URL,
                json = {
                    "model": LLM.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "logprobs": True,
                    "top_logprobs": LLM.TOP_LOGPROBS,
                    "keep_alive": LLM.OLLAMA_KEEP_ALIVE,
                    "options": LLM.CANDIDATE_GENERATION_TUNING_PARAMS[search_level]
                },
                timeout=LLM.OLLAMA_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result = response.json()
            response = result.get("response", "")
            logprobs = result.get("logprobs", [])
            logprob_results = LLM._aggregate_logprobs(logprobs, entry.length)

            formatted_output = ", ".join([f"{word} ({int(confidence)})" for word, confidence in logprob_results])
            logger.debug(f"[LOGPROBS RESULT] {formatted_output}")

            # Parse the LLM response and add answers that match the criteria to result set
            candidates: dict[str, Candidate] = {}
            for answer in response.split("\n"):
                answer = answer.strip()
                if not answer:
                    continue
                if len(answer) != entry.length:
                    continue
                if not answer.isalpha():
                    continue
                #if not matches_pattern(answer, entry.pattern):
                #    continue
                candidates[answer] = Candidate(
                    entry_id=entry.entry_id, 
                    answer=answer
                )

            # Add hints that match the criteria to the result set
            if entry.hints:
                for _, answer in entry.hints:
                    if len(answer) != entry.length:
                        continue
                    if not answer.isalpha():
                        continue
                    #if not matches_pattern(answer, entry.pattern):
                    #    continue
                    candidates[answer] = Candidate(
                        entry_id=entry.entry_id, 
                        answer=answer
                    )

            # Add logprob words that match the criteria to the result set
            for answer, confidence in logprob_results:
                if len(answer) != entry.length:
                    continue
                if not answer.isalpha():
                    continue
                #if not matches_pattern(answer, entry.pattern):
                #    continue
                candidates[answer] = Candidate(
                    entry_id=entry.entry_id, 
                    answer=answer,
                    logprob_confidence=confidence
                )

            logger.debug(f"[LLM GENERATE RESULT] Generated {len(candidates)} candidates: {', '.join(c.answer for c in candidates.values())}")
            return candidates
        except Exception as e:
            logger.error(f"[LLM GENERATE FATAL ERROR] Ollama generation query failed: {e}")
            raise


    @staticmethod
    def _create_prompt(entry: Entry, pattern: str, search_level: int, max_candidates: int = 0) -> str:
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
        :param search_level: Measures how "creative" we want the LLM to be with its answers.  See
            generate_candidates() for a fuller description.
            NOTE: For now, the max_search_level is 0 and this parameter is ignored.  We'll
            elaborate on more creative prompt generation later.
        :return: The prompt we will pass to the LLM.
        """
        clue: str = entry.clue
        length: int = entry.length
        hints: list[tuple[str,str]] | None = entry.hints

        matcher: str = f'[^{re.escape(".")}]'
        valid_pattern: bool = (re.search(matcher, pattern) is not None)
        if max_candidates == 0:
            max_candidates = LLM.MAX_CANDIDATES[search_level]

        prompt =  "TASK: Given a crossword clue and contextual hints, deduce CANDIDATE crossword answers.\n"
        prompt += "\nRULES:\n"
        prompt += "- A CANDIDATE is a potential answer deduced for the TARGET CLUE.\n"
        prompt += "- Many correct crossword answers are multi-word phrases.\n"
        prompt += "- Actively consider multi-word answers when deducing CANDIDATES.\n"
        prompt += "- Normalize each CANDIDATE by removing all spaces and punctuation and converting to upper case.\n"
        prompt += f"- A normalized CANDIDATE must be {length} characters.\n"
        prompt += "- DO NOT truncate or alter a CANDIDATE to fit the LENGTH, even if it seems like a good semantic fit.\n"
        prompt += "- DO alter a CANDIDATE's plurality or verb tense to match the TARGET CLUE.\n"
        if valid_pattern:
            prompt += f"The PATTERN of known letters is: {pattern}"
            prompt += (f"- A normalized CANDIDATE should match this PATTERN, where a period . is an unknown character.\n")
            prompt += "- When a PATTERN has only one or two unknown letters, focus on finding CANDIDATES that match the PATTERN exactly.\n"
        prompt += "- HINTS are past crossword clue-answer pairs semantically similar to the TARGET CLUE.\n"
        prompt += "- HINTS are unranked, and may be only loosely related to the TARGET CLUE.\n"
        prompt += "- HINTS do not provide an exhastive list of CANDIDATES, but they should be given additional weight.\n"
        #prompt += "- CANDIDATES may be inferred from general crossword knowledge and common idiomatic usage, even if not present in the HINTS.\n"
        #prompt += "- HINTS should be used to infer patterns or meanings.\n"
        if (search_level > 0):
            prompt += "- Generate creative and diverse CANDIDATES, even if unusual or speculative.\n"
        prompt += "OUTPUT FORMAT:\n"
        prompt += f"- Provide a list of up to {max_candidates} CANDIDATES.\n"
        prompt += "- Each CANDIDATE must be on its own line.\n"
        prompt += "- IMPORTANT: DO NOT provide any other text, ratings or scores.\n"
        if hints is not None and len(hints) > 0:
            prompt += "\nHINTS:\n"
            for hint_clue, hint_answer in hints:
                prompt += f"- REFERENCE CLUE: '{hint_clue}', REFERENCE ANSWER: '{hint_answer}'\n"
        if valid_pattern:
            prompt += (f"\nPATTERN: {pattern}\n")
        prompt += "\nTARGET CLUE: " + clue + "\n"
        prompt += "\nLENGTH: " + str(length) + "\n"

        return prompt


    @staticmethod
    def _aggregate_logprobs(
        logprobs_data: list[dict[str, Any]], 
        target_length: int,
        min_confidence: float = 0.0
    ) -> list[tuple[str, float]]:
        """
        Aggregates token-level log probabilities into word-level confidence scores.

        This function processes raw logprob data from an LLM, grouping sequential 
        tokens into candidate words based on delimiters (newlines, semicolons, etc.). 
        It normalizes candidates by converting them to uppercase and removing non-alphabetic 
        characters. If multiple instances of the same word (or truncated prefix) appear 
        in the data, their linear probabilities are summed to account for duplicates, 
        then capped at 100%.

        Args:
            logprobs_data: A list of dictionaries, where each dict contains a "token" 
                (str) and its associated "logprob" (float).
            target_length: The number of characters to truncate the normalized word 
                to for categorization/matching (useful for crossword slot constraints).
            min_confidence: The minimum confidence percentage (0.0 to 100.0) required 
                for a word to be included in the final results.

        Returns:
            A list of (word, confidence) tuples, where 'word' is the uppercase 
            alphabetic string of 'target_length' and 'confidence' is a float 
            between 0.0 and 100.0. The list is sorted by confidence in descending order.

        Note:
            The function uses a 'sentinel' pattern to ensure the final word in the 
            buffer is processed without redundant code blocks.
        """        
        word_probs: dict[str, float] = {} 
        current_tokens: list[str] = []
        current_logprob_sum: float = 0.0

        # Sentinel forces the loop to process the final word buffer
        sentinel: dict[str,str|float] = {"token": "\n", "logprob": 0.0}
        rejected_words: list[str] = []
        for entry in logprobs_data + [sentinel]:
            token: str = entry.get("token", "")
            logprob: float = entry.get("logprob", 0.0)

            is_delimiter = "\n" in token or "\\n" in token or token.strip() in {",", ";", ""}
            
            if is_delimiter:
                if current_tokens:
                    # Join tokens into one uppercase string
                    word = "".join(current_tokens).upper()
                    # Filter non-alpha characters and rejoin into one string
                    word = "".join(filter(str.isalpha, word))

                    # Add words of the proper length to the dictionary (or combine if it's already there)
                    if len(word) == target_length:
                        word_probs[word] = word_probs.get(word, 0.0) + math.exp(current_logprob_sum)
                    else:
                        rejected_words.append(word)

                current_tokens = []
                current_logprob_sum = 0.0
            else:
                if len("".join(current_tokens)) < target_length:
                    current_tokens.append(token.strip())
                    current_logprob_sum += logprob

        if rejected_words:
            logger.debug(f"[LOGPROBS] Rejecting words of wrong length: {", ".join(w for w in rejected_words)}")
                
        # Cap at 1.0 (100%) and apply min_confidence filter
        final_results = [
            (word, min(1.0, prob) * 100)
            for word, prob in word_probs.items()
            if (min(1.0, prob) * 100) >= min_confidence
        ]
        # Round confidence values to one decimal place
        final_results = [(word, round(conf, 1)) for word, conf in final_results]

        # Sort by confidence (index 1) descending
        return sorted(final_results, key=lambda x: x[1], reverse=True)


    @staticmethod
    def _score_candidates(
            entry: Entry,
            candidates: dict[str,Candidate],
            search_level: int
    ) -> list[Candidate]:
        """
        Score a list of candidate answers independently (evaluation phase).
        
        Each candidate is evaluated separately for how well it fits the clue and pattern,
        which improves calibration compared to scoring during generation.
        
        :param entry: The crossword entry
        :param answers: List of candidate answers to score
        :return: List of Candidate with confidence ratings
        """
        if not candidates:
            return []
        
        clue = entry.clue
        hints = entry.hints
        
        # Build a prompt for evaluating these specific candidates
        prompt = "TASK: Evaluate how well each CANDIDATE answer fits the CLUE and PATTERN.\n"
        prompt += "\nRULES:\n"
        prompt += "- For each CANDIDATE, determine if it is a plausible answer for the CLUE.\n"
        prompt += "- HINTS are provided for context but are not exhaustive.\n"
        prompt += "\nCONFIDENCE RUBRIC:\n"
        prompt += "- 90-100: Definitive answer; very confident match.\n"
        prompt += "- 80-89: Strong match with subtle interpretation.\n"
        prompt += "- 50-79: Plausible but less certain.\n"
        prompt += "- 0-50: Speculative or uncertain.\n"
        
        if hints:
            prompt += "\nHINTS (for context):\n"
            for hint_clue, hint_answer in hints:
                prompt += f"- '{hint_clue}' -> '{hint_answer}'\n"
        
        prompt += f"\nCLUE: {clue}\n"
        prompt += "\nCANDIDATES TO EVALUATE:\n"
        for candidate in candidates.values():
            prompt += f"- {candidate.answer}\n"
        
        prompt += "\nRESPONSE FORMAT:\n"
        prompt += "- For each candidate, provide: ANSWER | CONFIDENCE\n"
        prompt += "- ONLY provide ANSWER and CONFIDENCE, no other text.\n"
        prompt += "- Select a single value for CONFIDENCE, do not specify a range.\n"
        
        try:
            #logger.debug(f"[LLM SCORE] Scoring {len(candidates)} candidates for entry {entry.entry_id}, clue '{clue}'")
            response = requests.post(
                LLM.OLLAMA_URL,
                json = {
                    "model": LLM.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": LLM.OLLAMA_KEEP_ALIVE
                },
                timeout=LLM.OLLAMA_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "")
            
            # Parse scored candidates
            for part in output.split("\n"):
                part = part.strip()
                if "|" in part:
                    answer, conf = part.split("|", 1)
                    answer = normalize_candidate(answer)
                    try:
                        confidence = int(conf.strip())
                        confidence = max(0, min(confidence, 100))
                    except ValueError:
                        logger.error(f"[LLM SCORE ERROR] Unable to parse score for answer {part}, using default.")
                        confidence = LLM.DEFAULT_CONFIDENCE
                    exsting_candidate = candidates.get(answer)
                    if exsting_candidate is not None:
                        exsting_candidate.llm_confidence = float(confidence)
                    
            
            # Sort candidates by confidence descending
            sorted_candidates = sorted(candidates.values(), key=lambda c: c.llm_confidence, reverse=True)

            formatted_output = ", ".join([f"{c.answer} ({c.llm_confidence:.1f})" for c in sorted_candidates])
            logger.debug(f"[LLM SCORE RESULT] {formatted_output}")

            return sorted_candidates
        except Exception as e:
            logger.error(f"[LLM SCORE FATAL ERROR] {e}")
            raise


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
        # First check whether the answer is already in our candidate pool.  If so, it's obviously
        # a plausible answer.
        for candidate in entry.get_candidates():
            if answer == candidate.answer:
                return True

        #prompt: str = "Is the given ANSWER plausible for the given crossword CLUE?\n"
        #prompt += "IMPORTANT: An ANSWER may consist of multiple words.\n"
        #prompt += "You may add spaces to the ANSWER to make it into a phrase that satisfies the CLUE.\n\n"
        #prompt += (f"CLUE: {entry.clue}\n")
        #prompt += (f"LENGTH: {entry.length}\n")
        #prompt += (f"ANSWER: {answer}\n\n")
        #prompt += "Unless it is a proper noun or abbreviation, it must consist of valid words, spelled correctly.\n"
        #prompt += "Respond ONLY with the word Yes or No and no other text.\n"
        prompt: str = f"For the crossword puzzle clue '{entry.clue}', is {answer} a plausible answer?\n"
        prompt += "Respond only with Yes or No"

        try:
            logger.debug(f"[LLM VERIFY] Clue: '{entry.clue}' | Answer: '{answer}' | Length: {entry.length}")
            response = requests.post(
                LLM.OLLAMA_URL,
                json = {
                    "model": LLM.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": LLM.OLLAMA_KEEP_ALIVE
                },
                timeout=LLM.OLLAMA_TIMEOUT_SECONDS
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Manual test: print candidates from a live LLM call.
    clue = "Old-fashioned butter maker"
    answer = "CHURN"
    length = len(answer)
    grid: list[list[Cell]] = [[Cell(0, i) for i in range(length)]]
    entry = Entry("1A", clue, answer, grid, (0, 0), length)

    print(f"Clue: {entry.clue}")
    candidates = LLM.generate_candidates(entry, pattern=entry.pattern, search_level=0)
    print("Candidates:")
    for cand in candidates:
        print(f"- {cand.answer} ({cand.confidence})")
