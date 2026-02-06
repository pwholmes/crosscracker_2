from unittest.mock import Mock, patch
from src.llm import LLM
from src.model import Entry, Cell, ScoredCandidate


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
    
    with patch('src.llm.requests.post', side_effect=[mock_generation_response, mock_scoring_response]) as mock_post:
        candidates = LLM.generate_candidates(entry, widening_level=0)
        
        # Verify requests.post was called twice (generation + scoring)
        assert mock_post.call_count == 2
        
        # Verify the candidates were parsed correctly
        assert len(candidates) == 2
        assert candidates[0].answer == "CHURN"
        assert candidates[0].confidence == 95.0
        assert candidates[1].answer == "MIXER"
        assert candidates[1].confidence == 60.0


def test_generate_candidates_with_malformed_response():
    """Test that generate_candidates handles malformed responses gracefully."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "BADFORMAT TESTS 85 ANOTHER|invalid"
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.llm.requests.post', return_value=mock_response):
        candidates = LLM.generate_candidates(entry, widening_level=0)
        
        # Should return empty list or only valid candidates
        assert isinstance(candidates, list)
        assert all(isinstance(c, ScoredCandidate) for c in candidates)


def test_generate_candidates_with_network_error():
    """Test that generate_candidates handles network errors gracefully."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    with patch('src.llm.requests.post', side_effect=Exception("Network error")):
        candidates = LLM.generate_candidates(entry, widening_level=0)
        
        # Should return empty list on error
        assert candidates == []


def test_generate_candidates_respects_max_candidates():
    """Test that generate_candidates limits results to MAX_CANDIDATES."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    mock_response = Mock()
    # Provide more than MAX_CANDIDATES candidates
    mock_response.json.return_value = {
        "response": "TESTS|90\nTESTA|85\nTESTB|80\nTESTC|75\nTESTD|70\nTESTE|65"
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.llm.requests.post', return_value=mock_response):
        candidates = LLM.generate_candidates(entry, widening_level=0)
        
        # Should return only MAX_CANDIDATES candidates
        assert len(candidates) <= LLM.MAX_CANDIDATES
        assert candidates[0].answer == "TESTS"


def test_generate_candidates_filters_by_length():
    """Test that generate_candidates only returns answers matching entry length."""
    entry = create_test_entry("Test clue", "ABCD", 4)
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "ABCD|90\nTOOLONG|85\nABC|80\nOKAY|75"
    }
    mock_response.raise_for_status = Mock()
    
    with patch('src.llm.requests.post', return_value=mock_response):
        candidates = LLM.generate_candidates(entry, widening_level=0)
        
        # Should only return answers with length 4
        assert all(len(c.answer) == 4 for c in candidates)
        assert candidates[0].answer == "ABCD"
        assert candidates[1].answer == "OKAY"


def test_generate_candidates_with_hook():
    """Test that generate_candidates uses hook when set."""
    entry = create_test_entry("Test clue", "TESTS", 5)
    
    # Create a mock hook
    mock_hook = Mock(return_value=[
        ScoredCandidate(answer="HOOKA", confidence=100.0),
        ScoredCandidate(answer="HOOKB", confidence=90.0),
    ])
    
    # Set the hook
    LLM.set_generate_candidates_hook(mock_hook)
    
    try:
        with patch('src.llm.requests.post') as mock_post:
            candidates = LLM.generate_candidates(entry, widening_level=0)
            
            # Verify hook was called and requests.post was NOT called
            mock_hook.assert_called_once_with(entry, 0, LLM.MAX_CANDIDATES)
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
    
    with patch('src.llm.requests.post', return_value=mock_response) as mock_post:
        result = LLM.verify_answer(entry, answer)
        
        # Verify requests.post was called with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs['json']['model'] == LLM.MODEL_NAME
        assert call_args.kwargs['json']['stream'] is False
        assert 'prompt' in call_args.kwargs['json']
        
        # Verify the prompt contains the clue and answer
        prompt = call_args.kwargs['json']['prompt']
        assert clue in prompt
        assert answer in prompt
        assert "Yes or No" in prompt
        
        # Verify the result is True
        assert result is True


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
    
    with patch('src.llm.requests.post', return_value=mock_response):
        result = LLM.verify_answer(entry, answer)
        
        # Verify the result is False
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
        
        with patch('src.llm.requests.post', return_value=mock_response):
            result = LLM.verify_answer(entry, answer)
            
            # Should return False for any response other than exactly "Yes"
            assert result is False, f"Expected False for response '{response_text}'"


def test_verify_answer_handles_network_error():
    """Test that verify_answer returns False when network error occurs."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    with patch('src.llm.requests.post', side_effect=Exception("Network error")):
        result = LLM.verify_answer(entry, answer)
        
        # Should return False on error
        assert result is False


def test_verify_answer_handles_timeout():
    """Test that verify_answer handles timeout errors gracefully."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    with patch('src.llm.requests.post', side_effect=TimeoutError("Request timed out")):
        result = LLM.verify_answer(entry, answer)
        
        # Should return False on timeout
        assert result is False


def test_verify_answer_handles_http_error():
    """Test that verify_answer handles HTTP errors gracefully."""
    clue = "Test clue"
    answer = "TESTS"
    entry = create_test_entry(clue, answer, len(answer))
    
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 500")
    
    with patch('src.llm.requests.post', return_value=mock_response):
        result = LLM.verify_answer(entry, answer)
        
        # Should return False on HTTP error
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
    
    with patch('src.llm.requests.post', return_value=mock_response) as mock_post:
        LLM.verify_answer(entry, answer)
        
        # Verify timeout parameter is set to 30 seconds
        call_args = mock_post.call_args
        assert call_args.kwargs['timeout'] == 30

