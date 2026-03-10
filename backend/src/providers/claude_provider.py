"""
Claude LLM provider implementation via Anthropic API.

Handles communication with Claude models through the Anthropic SDK.
Since Claude doesn't expose logprobs like Ollama, we rely on LLM-scored confidence only.
"""

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

from llm_provider import LLMProvider
from model import Entry, Candidate
from anthropic import Anthropic, AsyncAnthropic

logger = logging.getLogger("src.providers.claude")


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from Anthropic API response message.
    
    Handles the various content block types that the API may return.
    """
    if not response.content or len(response.content) == 0:
        return ""
    
    content_block = response.content[0]
    
    # Check if it's a TextBlock with text attribute
    if hasattr(content_block, 'text'):
        text_attr = getattr(content_block, 'text', None)
        if isinstance(text_attr, str):
            return text_attr
    
    # Check if it's a dict-like object  
    if isinstance(content_block, dict):
        # Type-safe extraction from dict
        text_value = content_block.get("text")  # type: ignore
        if isinstance(text_value, str):
            return text_value
    
    return ""


class ClaudeProvider(LLMProvider):
    """
    LLM provider implementation for Anthropic Claude models.
    """
    
    # Configuration for candidate generation behavior
    DEFAULT_CONFIDENCE = 30.0
    MAX_SEARCH_LEVEL: int = 2
    MAX_CANDIDATES: list[int] = [7, 12, 15]
    
    def __init__(self):
        """Initialize Claude provider from environment variables."""
        load_dotenv()
        api_key = os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY environment variable is not set")
        
        self.model_name: str = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        logger.debug(f"Initialized Claude provider: model={self.model_name}")

    def generate_candidates(
        self,
        entry: Entry,
        pattern: str,
        search_level: int
    ) -> dict[str, Candidate]:
        """
        Generate candidate answers for a clue using Claude.
        """
        prompt: str = self._create_prompt(entry, pattern, search_level)
        
        try:
            logger.debug(f"[LLM GENERATE] Entry: {entry.entry_id} | Clue: '{entry.clue}' | Length: {entry.length} | Pattern: {entry.pattern} | Search level {search_level}")
            if entry.hints:
                hints_str = ", ".join(f"'{clue}'->{answer}" for clue, answer, _ in entry.hints)
                logger.debug(f"[LLM GENERATE HINTS] {hints_str}")

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = _extract_text_from_response(response)

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

            logger.debug(f"[LLM GENERATE RESULT] Generated {len(candidates)} candidates: {', '.join(c.answer for c in candidates.values())}")
            return candidates
        except Exception as e:
            logger.error(f"[LLM GENERATE FATAL ERROR] Claude generation query failed: {e}")
            raise

    def score_candidates(
        self,
        entry: Entry,
        candidates: dict[str, Candidate],
        search_level: int
    ) -> list[Candidate]:
        """
        Score a set of candidate answers using Claude.
        """
        if not candidates:
            return []
        
        clue = entry.clue
        hints = entry.hints
        
        # Build a prompt for evaluating these specific candidates
        prompt = "TASK: Evaluate how well each CANDIDATE answer fits the CLUE.\n"
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
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            output = _extract_text_from_response(response)
            
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
        Verify if an answer is plausible for a clue using Claude.
        """
        # First check whether the answer is already in our candidate pool
        for candidate in entry.get_candidates():
            if answer == candidate.answer:
                return True

        prompt: str = f"For the crossword puzzle clue '{entry.clue}', is {answer} a plausible answer?\n"
        prompt += "Respond only with Yes or No"

        try:
            logger.debug(f"[LLM VERIFY] Clue: '{entry.clue}' | Answer: '{answer}' | Length: {entry.length}")
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            output = _extract_text_from_response(response)
            is_valid: bool = output.strip() == "Yes"
            logger.debug(f"[LLM VERIFY] Result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Claude query failed: {e}")
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
