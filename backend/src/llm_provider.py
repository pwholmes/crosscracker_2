"""
Abstract base class for LLM provider implementations.

This module defines the interface that all LLM providers must implement,
allowing CrossCracker to work with different LLM backends transparently.
"""

from abc import ABC, abstractmethod
from model import Entry, Candidate


class LLMProvider(ABC):
    """
    Abstract base class for LLM backend providers.
    
    Each provider implementation should handle:
    - Candidate generation from crossword clues
    - Candidate scoring/confidence evaluation
    - Answer verification
    """

    @abstractmethod
    def generate_candidates(
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
        pass

    @abstractmethod
    def score_candidates(
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
        pass

    @abstractmethod
    def verify_answer(
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
        pass

    def normalize_candidate(self, answer: str) -> str:
        """Normalize a candidate answer: capitalize and remove spaces, punctuation, and digits."""
        return ''.join(ch.upper() for ch in answer if ch.isalpha())


