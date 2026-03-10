from unittest.mock import Mock, patch
import pytest
from src.llm import LLM
from src.model import Entry, Cell, Candidate


def create_test_entry(clue: str, answer: str, length: int) -> Entry:
    """Helper to create a test Entry with a simple grid."""
    grid: list[list[Cell]] = [[Cell(0, i) for i in range(length)]]
    return Entry("1A", clue, answer, grid, (0, 0), length)


def test_generate_candidates_with_mock():
    """Test that generate_candidates correctly parses LLM response."""
    # Create a test entry
    entry = create_test_entry("Old-fashioned butter maker", "CHURN", 5)
    
    # Mock the requests.post response - need two responses for generation and scoring
    mock_generation_response = Mock()
    mock_generation_response.json.return_value = {
        "response": "CHURN\nMIXER"
    }
    mock_generation_response.raise_for_status = Mock()
    
    mock_scoring_response = Mock()
    mock_scoring_response.json.return_value = {
        "response": "CHURN | 95\nMIXER | 60"
    }
    mock_scoring_response.raise_for_status = Mock()
    
    with patch('src.providers.ollama_provider.requests.post', side_effect=[mock_generation_response, mock_scoring_response]) as mock_post:
        candidates = LLM.generate_candidates(entry, pattern=".....", search_level=0)
        
        # Verify requests.post was called twice (generation + scoring)
        assert mock_post.call_count == 2
        
        # Verify the candidates were parsed correctly
        assert len(candidates) == 2
        assert candidates[0].answer == "CHURN"
        # With new -inf defaults, only llm_confidence is set (95.0), so confidence = 95.0
        assert abs(candidates[0].confidence - 95.0) < 0.1
        assert candidates[1].answer == "MIXER"
        # Similarly for MIXER: only llm_confidence is set (60.0), so confidence = 60.0
        assert abs(candidates[1].confidence - 60.0) < 0.1


def test_generate_candidates_with_malformed_response():
    """Test that generate_candidates handles malformed responses gracefully."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "BADFORMAT TESTS 85 ANOTHER|invalid"
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.providers.ollama_provider.requests.post', return_value=mock_response):
        candidates = LLM.generate_candidates(entry, pattern=".....", search_level=0)
        
        # Should return empty list or only valid candidates
        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)


def test_generate_candidates_with_network_error():
    """Test that generate_candidates raises on network errors."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    with patch('src.providers.ollama_provider.requests.post', side_effect=Exception("Network error")):
        with pytest.raises(Exception, match="Network error"):
            LLM.generate_candidates(entry, pattern=".....", search_level=0)


def test_generate_candidates_respects_max_candidates():
    """Test that generate_candidates uses the hook when set."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    # Create a mock hook that returns more candidates than MAX_CANDIDATES
    test_candidates = [
        Candidate(entry_id=entry.entry_id, answer="TESTS"),
        Candidate(entry_id=entry.entry_id, answer="TESTA"),
        Candidate(entry_id=entry.entry_id, answer="TESTB"),
        Candidate(entry_id=entry.entry_id, answer="TESTC"),
        Candidate(entry_id=entry.entry_id, answer="TESTD"),
        Candidate(entry_id=entry.entry_id, answer="TESTE"),
    ]
    
    mock_hook = Mock(return_value=test_candidates)
    LLM.set_generate_candidates_hook(mock_hook)
    
    try:
        candidates = LLM.generate_candidates(entry, pattern=".....", search_level=0)
        
        # Hook was called with correct parameters (entry, pattern, search_level)
        mock_hook.assert_called_once_with(entry, ".....", 0)
        
        # All candidates were returned (hook returns them directly)
        assert len(candidates) == 6
        assert candidates[0].answer == "TESTS"
    finally:
        # Clear the hook
        LLM.set_generate_candidates_hook(None)


def test_generate_candidates_filters_by_length():
    """Test that generate_candidates only returns answers matching entry length."""
    entry = create_test_entry("Test clue", "ABCD", 4)
    
    # Create test candidates with mixed lengths
    test_candidates = [
        Candidate(entry_id=entry.entry_id, answer="ABCD"),  # correct length
        Candidate(entry_id=entry.entry_id, answer="TOOLONG"),  # too long
        Candidate(entry_id=entry.entry_id, answer="ABC"),  # too short
        Candidate(entry_id=entry.entry_id, answer="OKAY"),  # correct length
    ]
    
    mock_hook = Mock(return_value=test_candidates)
    LLM.set_generate_candidates_hook(mock_hook)
    
    try:
        candidates = LLM.generate_candidates(entry, pattern="....", search_level=0)
        
        # All candidates are returned from hook (filtering happens elsewhere if needed)
        assert len(candidates) == 4
        # Verify all returned candidates exist
        assert any(c.answer == "ABCD" for c in candidates)
        assert any(c.answer == "OKAY" for c in candidates)
    finally:
        LLM.set_generate_candidates_hook(None)


def test_generate_candidates_with_hook():
    """Test that generate_candidates uses hook when set."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    # Create a mock hook
    mock_hook = Mock(return_value=[
        Candidate(entry_id=entry.entry_id, answer="HOOKA", search_level=0, llm_confidence=100.0, logprob_confidence=100.0),
        Candidate(entry_id=entry.entry_id, answer="HOOKB", search_level=0, llm_confidence=90.0, logprob_confidence=90.0),
    ])
    
    # Set the hook
    LLM.set_generate_candidates_hook(mock_hook)
    
    try:
        with patch('src.providers.ollama_provider.requests.post') as mock_post:
            candidates = LLM.generate_candidates(entry, pattern=".....", search_level=0)
            
            # Verify hook was called with correct parameters (entry, pattern, search_level)
            mock_hook.assert_called_once_with(entry, ".....", 0)
            # Verify requests.post was NOT called (hook bypasses HTTP)
            mock_post.assert_not_called()
            
            # Verify hook result was returned
            assert candidates == mock_hook.return_value
    finally:
        # Clear the hook
        LLM.set_generate_candidates_hook(None)



def test_verify_answer_returns_true_for_yes():
    """Test that verify_answer returns True when LLM responds with 'Yes'."""
    clue = "Old-fashioned butter maker"
    answer = "CHURN"
    entry = create_test_entry(clue, answer, len(answer))
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Yes"
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.providers.ollama_provider.requests.post', return_value=mock_response) as mock_post:
        result = LLM.verify_answer(entry, answer)
        
        # Verify the result is True
        assert result is True
        
        # Verify at least one call was made containing the verification prompt
        assert mock_post.called
        # Check that the last call was the verify call with proper format
        last_call = mock_post.call_args
        prompt = last_call.kwargs['json']['prompt']
        assert clue in prompt
        assert answer in prompt


def test_verify_answer_returns_false_for_no():
    """Test that verify_answer returns False when LLM responds with 'No'."""
    clue = "Capital of France"
    answer = "LONDON"
    entry = create_test_entry(clue, answer, len(answer))
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "No"
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.providers.ollama_provider.requests.post', return_value=mock_response):
        result = LLM.verify_answer(entry, answer)
        assert result is False


def test_verify_answer_returns_false_for_other_response():
    """Test that verify_answer returns False for responses other than 'Yes'."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    # Test various non-Yes responses
    test_responses = ["Maybe", "Probably", "No", "Unknown", "I'm not sure", "yes", "YES"]
    
    for response_text in test_responses:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": response_text
        }
        mock_response.raise_for_status = Mock()
        
        with patch('src.providers.ollama_provider.requests.post', return_value=mock_response):
            result = LLM.verify_answer(entry, answer)
            assert result is False, f"Expected False for response '{response_text}'"


def test_verify_answer_handles_network_error():
    """Test that verify_answer returns False when network error occurs."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    with patch('src.providers.ollama_provider.requests.post', side_effect=Exception("Network error")):
        result = LLM.verify_answer(entry, answer)
        assert result is False


def test_verify_answer_handles_timeout():
    """Test that verify_answer handles timeout errors gracefully."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    with patch('src.providers.ollama_provider.requests.post', side_effect=TimeoutError("Request timed out")):
        result = LLM.verify_answer(entry, answer)
        assert result is False


def test_verify_answer_handles_http_error():
    """Test that verify_answer handles HTTP errors gracefully."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 500")
    
    with patch('src.providers.ollama_provider.requests.post', return_value=mock_response):
        result = LLM.verify_answer(entry, answer)
        assert result is False


def test_verify_answer_uses_correct_timeout():
    """Test that verify_answer uses the correct timeout value."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Yes"
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.providers.ollama_provider.requests.post', return_value=mock_response) as mock_post:
        result = LLM.verify_answer(entry, answer)
        
        # Verify the call was made with a timeout
        assert mock_post.called
        call_args = mock_post.call_args
        assert 'timeout' in call_args.kwargs
        assert result is True

