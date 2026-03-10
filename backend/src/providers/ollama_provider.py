"""
Ollama LLM provider implementation.

Handles communication with local Ollama instances running LLMs like llama3.1:8b.
"""

import requests
import logging
import re
import os
import math
from typing import Any
from dotenv import load_dotenv

from llm_provider import LLMProvider
from model import Entry, Candidate

logger = logging.getLogger("src.providers.ollama")


class OllamaProvider(LLMProvider):
    """
    LLM provider implementation for local Ollama instances.
    """
    
    # Configuration for candidate generation behavior
    DEFAULT_CONFIDENCE = 30.0
    MAX_SEARCH_LEVEL: int = 2
    MAX_CANDIDATES: list[int] = [7, 12, 15]
    CANDIDATE_GENERATION_TUNING_PARAMS: list[dict[str, float | int]] = [
        {"temperature": 0.25, "top_p": 0.8, "top_k": 10},
        {"temperature": 0.70, "top_p": 0.9, "top_k": 30},
        {"temperature": 1.00, "top_p": 0.95, "top_k": 60}
    ]
    TIMEOUT_SECONDS = 120
    KEEP_ALIVE = "30m"
    TOP_LOGPROBS = 5

    def __init__(self):
        """Initialize Ollama provider from environment variables."""
        load_dotenv()
        self.model_name: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        logger.debug(f"Initialized Ollama provider: model={self.model_name}, url={self.url}")

    def generate_candidates(
        self,
        entry: Entry,
        pattern: str,
        search_level: int
    ) -> dict[str, Candidate]:
        """
        Generate candidate answers for a clue using Ollama.
        """
        prompt: str = self._create_prompt(entry, pattern, search_level)
        
        try:
            logger.debug(f"[LLM GENERATE] Entry: {entry.entry_id} | Clue: '{entry.clue}' | Length: {entry.length} | Pattern: {entry.pattern} | Search level {search_level}")
            if entry.hints:
                hints_str = ", ".join(f"'{clue}'->{answer}" for clue, answer, _ in entry.hints)
                logger.debug(f"[LLM GENERATE HINTS] {hints_str}")

            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "logprobs": True,
                    "top_logprobs": self.TOP_LOGPROBS,
                    "keep_alive": self.KEEP_ALIVE,
                    "options": self.CANDIDATE_GENERATION_TUNING_PARAMS[search_level]
                },
                timeout=self.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            response_text: str = result.get("response", "")
            logprobs: list[dict[str, float]] | Any = result.get("logprobs", [])
            logprob_results: list[tuple[str, float]] = self._aggregate_logprobs(logprobs, entry.length)

            formatted_output = ", ".join([f"{word} ({int(confidence)})" for word, confidence in logprob_results])
            logger.debug(f"[LOGPROBS RESULT] {formatted_output}")

            # Parse the LLM response and add answers that match the criteria
            candidates: dict[str, Candidate] = {}
            for answer in response_text.split("\n"):
                answer = answer.strip()
                if not answer:
                    continue
                if len(answer) != entry.length:
                    continue
                if not answer.isalpha():
                    continue
                candidates[answer] = Candidate(
                    entry_id=entry.entry_id, 
                    answer=answer
                )

            # Add hints that match the criteria
            if entry.hints:
                for _, answer, distance in entry.hints:
                    if len(answer) != entry.length:
                        continue
                    if not answer.isalpha():
                        continue
                    candidates[answer] = Candidate(
                        entry_id=entry.entry_id, 
                        answer=answer,
                        llm_confidence=1/(1+distance)
                    )

            # Add logprob words that match the criteria
            for answer, confidence in logprob_results:
                if len(answer) != entry.length:
                    continue
                if not answer.isalpha():
                    continue
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

    def score_candidates(
        self,
        entry: Entry,
        candidates: dict[str, Candidate],
        search_level: int
    ) -> list[Candidate]:
        """
        Score a set of candidate answers using Ollama.
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
            for hint_clue, hint_answer, _ in hints:
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
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.KEEP_ALIVE
                },
                timeout=self.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result = response.json()
            output = result.get("response", "")
            
            # Parse scored candidates
            for part in output.split("\n"):
                part = part.strip()
                if "|" in part:
                    answer, conf = part.split("|", 1)
                    answer = self.normalize_candidate(answer)
                    try:
                        confidence = int(conf.strip())
                        confidence = max(0, min(confidence, 100))
                    except ValueError:
                        logger.error(f"[LLM SCORE ERROR] Unable to parse score for answer {part}, using default.")
                        confidence = self.DEFAULT_CONFIDENCE
                    existing_candidate = candidates.get(answer)
                    if existing_candidate is not None:
                        existing_candidate.llm_confidence = float(confidence)
            
            # Sort candidates by confidence descending
            sorted_candidates = sorted(candidates.values(), key=lambda c: c.llm_confidence, reverse=True)

            formatted_output = ", ".join([f"{c.answer} ({c.llm_confidence:.1f})" for c in sorted_candidates])
            logger.debug(f"[LLM SCORE RESULT] {formatted_output}")

            return sorted_candidates
        except Exception as e:
            logger.error(f"[LLM SCORE FATAL ERROR] {e}")
            raise

    def verify_answer(
        self,
        entry: Entry,
        answer: str
    ) -> bool:
        """
        Verify if an answer is plausible for a clue using Ollama.
        """
        try:
            # First check whether the answer is already in our candidate pool
            for candidate in entry.get_candidates():
                if answer == candidate.answer:
                    return True
        except Exception as e:
            logger.debug(f"[LLM VERIFY] Could not retrieve candidates: {e}")
            # Continue to LLM verification even if candidate retrieval fails

        prompt: str = f"For the crossword puzzle clue '{entry.clue}', is {answer} a plausible answer?\n"
        prompt += "Respond only with Yes or No"

        try:
            logger.debug(f"[LLM VERIFY] Clue: '{entry.clue}' | Answer: '{answer}' | Length: {entry.length}")
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.KEEP_ALIVE
                },
                timeout=self.TIMEOUT_SECONDS
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

    def _create_prompt(self, entry: Entry, pattern: str, search_level: int, max_candidates: int = 0) -> str:
        """Create a prompt for candidate generation."""
        clue: str = entry.clue
        length: int = entry.length
        hints: list[tuple[str, str, float]] | None = entry.hints

        matcher: str = f'[^{re.escape(".")}]'
        valid_pattern: bool = (re.search(matcher, pattern) is not None)
        if max_candidates == 0:
            max_candidates = self.MAX_CANDIDATES[search_level]

        prompt = "TASK: Given a crossword clue and contextual hints, deduce CANDIDATE crossword answers.\n"
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
        prompt += "- HINTS do not provide an exhaustive list of CANDIDATES, but they should be given additional weight.\n"
        if search_level > 0:
            prompt += "- Generate creative and diverse CANDIDATES, even if unusual or speculative.\n"
        prompt += "OUTPUT FORMAT:\n"
        prompt += f"- Provide a list of up to {max_candidates} CANDIDATES.\n"
        prompt += "- Each CANDIDATE must be on its own line.\n"
        prompt += "- IMPORTANT: DO NOT provide any other text, ratings or scores.\n"
        if hints is not None and len(hints) > 0:
            prompt += "\nHINTS:\n"
            for hint_clue, hint_answer, _ in hints:
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
        """Aggregate token-level log probabilities into word-level confidence scores."""
        word_probs: dict[str, float] = {} 
        current_tokens: list[str] = []
        current_logprob_sum: float = 0.0

        # Sentinel forces the loop to process the final word buffer
        sentinel: dict[str, Any] = {"token": "\n", "logprob": 0.0}
        rejected_words: list[str] = []
        for entry in logprobs_data + [sentinel]:
            token: str = str(entry.get("token", ""))
            logprob_val: Any = entry.get("logprob", 0.0)
            logprob: float = float(logprob_val) if isinstance(logprob_val, (int, float)) else 0.0

            is_delimiter = "\n" in token or "\\n" in token or token.strip() in {",", ";", ""}
            
            if is_delimiter:
                if current_tokens:
                    word = "".join(current_tokens).upper()
                    word = "".join(filter(str.isalpha, word))

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
            logger.debug(f"[LOGPROBS] Rejecting words of wrong length: {', '.join(w for w in rejected_words)}")
                
        # Cap at 1.0 (100%) and apply min_confidence filter
        final_results = [
            (word, min(1.0, prob) * 100)
            for word, prob in word_probs.items()
            if (min(1.0, prob) * 100) >= min_confidence
        ]
        final_results = [(word, round(conf, 1)) for word, conf in final_results]

        return sorted(final_results, key=lambda x: x[1], reverse=True)
