"""
Abstract base class for LLM provider implementations.

This module defines the interface that all LLM providers must implement,
allowing CrossCracker to work with different LLM backends transparently.
"""

from abc import ABC, abstractmethod
import re
from typing import Any
from model import Entry, Candidate


class LLMProvider(ABC):
    """
    Abstract base class for LLM backend providers.
    
    Each provider implementation should handle:
    - Candidate generation from crossword clues
    - Candidate scoring/confidence evaluation
    - Answer verification
    """

    def provider_generate_candidates(
        self,
        entry: Entry,
        pattern: str,
        search_level: int
    ) -> dict[str, Candidate]:
        """
        Generate candidate answers for a crossword clue.
        
        :param entry: The crossword entry with clue, hints, length, etc.
        :param pattern: Pattern of known letters (e.g., "A..B." where . is unknown)
        :param search_level: How creative to be (0=conservative, higher=more speculative)
        :return: Dict of {answer: Candidate} with answers as keys
        """
        prompt = self._create_generation_prompt(entry, pattern, search_level)
        
        response_text, raw_response = self._request_generation(prompt, search_level, entry)

        candidates = self._extract_candidates_from_response_text(entry, response_text)
        self._add_hint_candidates(entry, candidates)
        self._add_provider_candidates(entry, candidates, raw_response, search_level)

        return candidates


    def provider_score_candidates(
        self,
        entry: Entry,
        candidates: dict[str, Candidate],
        search_level: int
    ) -> list[Candidate]:
        """
        Score a set of candidate answers for how well they fit the clue.
        
        Updates the llm_confidence field of each candidate.
        
        :param entry: The crossword entry
        :param candidates: Dict of candidates to score
        :param search_level: Search level context
        :return: Sorted list of candidates by confidence (descending)
        """
        if not candidates:
            return []

        prompt = self._create_scoring_prompt(
            entry,
            candidates,
            include_pattern_in_task=self._include_pattern_in_scoring_task()
        )
        output = self._request_scoring(prompt, search_level, entry)
        self._apply_scored_confidences(output, candidates)

        return self._sort_candidates_by_confidence(candidates)


    def provider_verify_answer(
        self,
        entry: Entry,
        answer: str
    ) -> bool:
        """
        Verify if a given answer is plausible for a crossword clue.
        
        Used when a crossing clue fills in the final letter(s) of an entry,
        to validate the resulting word makes sense for the original clue.
        
        :param entry: The crossword entry
        :param answer: The answer to verify
        :return: True if plausible, False otherwise
        """
        try:
            # First check whether the answer is already in our candidate pool.
            for candidate in entry.get_candidates():
                if answer == candidate.answer:
                    return True
        except Exception:
            # Continue to LLM verification if candidate retrieval fails.
            pass

        prompt: str = f"For the crossword puzzle clue '{entry.clue}', is {answer} a plausible answer?\n"
        prompt += "Respond only with Yes or No"

        try:
            output = self._request_verify(prompt, entry)
            return output.strip() == "Yes"
        except Exception:
            return False


    @abstractmethod
    def _request_generation(self, prompt: str, search_level: int, entry: Entry) -> tuple[str, Any]:
        """Issue a generation request to the provider and return (response_text, raw_response)."""
        pass

    @abstractmethod
    def _request_scoring(self, prompt: str, search_level: int, entry: Entry) -> str:
        """Issue a scoring request to the provider and return raw text output."""
        pass

    @abstractmethod
    def _request_verify(self, prompt: str, entry: Entry) -> str:
        """Issue a verification request to the provider and return raw text output."""
        pass

    def _include_pattern_in_scoring_task(self) -> bool:
        """Whether scoring prompts should include PATTERN in the task description."""
        return False

    def _add_provider_candidates(
        self,
        entry: Entry,
        candidates: dict[str, Candidate],
        raw_response: Any,
        search_level: int
    ) -> None:
        """Optional provider-specific candidate augmentation (e.g., logprob-based candidates)."""
        return None

    def _normalize_candidate(self, answer: str) -> str:
        """Normalize a candidate answer: capitalize and remove spaces, punctuation, and digits."""
        return ''.join(ch.upper() for ch in answer if ch.isalpha())

    def _create_generation_prompt(self, entry: Entry, pattern: str, search_level: int, max_candidates: int = 0) -> str:
        """Create a prompt for candidate generation."""
        from llm import LLM

        clue: str = entry.clue
        length: int = entry.length
        hints: list[tuple[str, str, float]] | None = entry.hints

        matcher: str = f'[^{re.escape(".")}]'
        valid_pattern: bool = (re.search(matcher, pattern) is not None)
        if max_candidates == 0:
            max_candidates = LLM.MAX_CANDIDATES[search_level]

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
            prompt += "- A normalized CANDIDATE should match this PATTERN, where a period . is an unknown character.\n"
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

    def _create_scoring_prompt(
        self,
        entry: Entry,
        candidates: dict[str, Candidate],
        include_pattern_in_task: bool = False
    ) -> str:
        """Create a prompt for scoring candidate answers against a clue."""
        clue = entry.clue
        hints = entry.hints

        task_target = "CLUE and PATTERN" if include_pattern_in_task else "CLUE"
        prompt = f"TASK: Evaluate how well each CANDIDATE answer fits the {task_target}.\n"
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

        return prompt

    def _extract_candidates_from_response_text(self, entry: Entry, response_text: str) -> dict[str, Candidate]:
        """Parse newline-delimited candidate answers from an LLM response."""
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
        return candidates

    def _add_hint_candidates(self, entry: Entry, candidates: dict[str, Candidate]) -> None:
        """Merge valid hint answers into a candidate map with distance-based confidence."""
        if entry.hints:
            for _, answer, distance in entry.hints:
                if len(answer) != entry.length:
                    continue
                if not answer.isalpha():
                    continue
                candidates[answer] = Candidate(
                    entry_id=entry.entry_id,
                    answer=answer,
                    llm_confidence=1 / (1 + distance)
                )

    def _apply_scored_confidences(self, output: str, candidates: dict[str, Candidate]) -> None:
        """Parse ANSWER | CONFIDENCE lines and update candidate confidence scores in-place."""
        from llm import LLM

        for part in output.split("\n"):
            part = part.strip()
            if "|" not in part:
                continue

            answer, conf = part.split("|", 1)
            answer = self._normalize_candidate(answer)
            try:
                confidence = int(conf.strip())
                confidence = max(0, min(confidence, 100))
            except ValueError:
                confidence = LLM.DEFAULT_CONFIDENCE

            existing_candidate = candidates.get(answer)
            if existing_candidate is not None:
                existing_candidate.llm_confidence = float(confidence)

    def _sort_candidates_by_confidence(self, candidates: dict[str, Candidate]) -> list[Candidate]:
        """Sort candidates by llm_confidence descending."""
        return sorted(candidates.values(), key=lambda c: c.llm_confidence, reverse=True)

    def _format_scored_candidates(self, sorted_candidates: list[Candidate]) -> str:
        """Format scored candidates for debug logging."""
        return ", ".join(f"{c.answer} ({c.llm_confidence:.1f})" for c in sorted_candidates)


