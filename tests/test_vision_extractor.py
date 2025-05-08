import pytest
import base64
import json
import requests
from unittest.mock import MagicMock, patch # pytest-mock provides the mocker fixture, but unittest.mock.patch is also commonly used

# Make sure src directory is in PYTHONPATH for imports
# This is often handled by pytest configuration (e.g. conftest.py or pytest.ini) or IDE settings.
# For simplicity here, if running this file directly or if src is not found,
# ensure your environment is set up. Pytest usually handles this fine.

from src.vision_extractor import extract_invoice_data
# We also need to access the config variables as the vision_extractor imports them directly
# For testing, we can patch these config values if needed, or ensure they are set minimally for tests.
# Let's assume src.config will be available or we mock its usage within extract_invoice_data if it causes issues during isolated testing.

# Dummy 1x1 transparent PNG image bytes for testing
DUMMY_IMAGE_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")

@pytest.fixture
def mock_openai_api_key(mocker):
    mocker.patch('src.vision_extractor.OPENAI_API_KEY', 'fake_openai_key')
    mocker.patch('src.vision_extractor.OPENAI_MODEL', 'gpt-4o-test')
    mocker.patch('src.vision_extractor.OPENAI_VISION_ENDPOINT', 'https://fake.openai.endpoint/v1/chat/completions')

@pytest.fixture
def mock_gemini_api_key(mocker):
    mocker.patch('src.vision_extractor.GEMINI_API_KEY', 'fake_gemini_key')
    mocker.patch('src.vision_extractor.GEMINI_MODEL', 'gemini-1.5-pro-test')
    mocker.patch('src.vision_extractor.GEMINI_VISION_ENDPOINT', 'https://fake.gemini.endpoint/v1beta/models/gemini-pro-vision:generateContent')


# --- Test Cases --- 

def test_extract_invoice_data_success_openai(mocker, mock_openai_api_key):
    """Test successful data extraction using OpenAI mock."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # The actual content returned by OpenAI API after it processes the image and prompt
    # This content itself should be a JSON string that our function then parses.
    api_content_json_string = json.dumps({
        "invoice_no": "INV-123",
        "date": "2023-10-26",
        "vendor": "Test Vendor Inc.",
        "total_amount": "150.75"
    })
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": api_content_json_string
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)

    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=True)

    assert result is not None
    assert result['invoice_no'] == "INV-123"
    assert result['date'] == "2023-10-26"
    assert result['vendor'] == "Test Vendor Inc."
    assert result['total_amount'] == 150.75 # Check for float conversion
    requests.post.assert_called_once()


def test_extract_invoice_data_success_openai_with_markdown_wrapper(mocker, mock_openai_api_key):
    """Test successful extraction when OpenAI returns JSON wrapped in markdown."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    api_content_json_string = json.dumps({
        "invoice_no": "INV-789",
        "date": "2024-01-15",
        "vendor": "Markdown Co.",
        "total_amount": "99.00"
    })
    # Simulate OpenAI wrapping the JSON in markdown code block
    wrapped_content = f"```json\n{api_content_json_string}\n```"
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": wrapped_content
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)
    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=True)
    assert result is not None
    assert result['invoice_no'] == "INV-789"
    assert result['total_amount'] == 99.00


def test_extract_invoice_data_success_gemini(mocker, mock_gemini_api_key):
    """Test successful data extraction using Gemini mock."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    api_content_json_string = json.dumps({
        "invoice_no": "GEM-456",
        "date": "2023-11-15",
        "vendor": "Gemini Supplies",
        "total_amount": "200.50"
    })
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": api_content_json_string}]
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)

    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=False)

    assert result is not None
    assert result['invoice_no'] == "GEM-456"
    assert result['date'] == "2023-11-15"
    assert result['vendor'] == "Gemini Supplies"
    assert result['total_amount'] == 200.50
    requests.post.assert_called_once()


def test_extract_invoice_data_api_http_error(mocker, mock_openai_api_key):
    """Test handling of HTTP error from the API."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError("API Error", response=mock_response))
    mocker.patch('requests.post', return_value=mock_response)

    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=True)
    assert result is None


def test_extract_invoice_data_network_error(mocker, mock_openai_api_key):
    """Test handling of a requests.exceptions.RequestException (e.g., network issue)."""
    mocker.patch('requests.post', side_effect=requests.exceptions.Timeout("Connection timed out"))

    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=True)
    assert result is None


def test_extract_invoice_data_malformed_json_content_openai(mocker, mock_openai_api_key):
    """Test handling of malformed JSON string within a successful API response (OpenAI)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    malformed_json_string = "{\"invoice_no\": \"INV-BAD\", \"date\": \"2023-10-27\", vendor: \"Malformed JSON" # Missing quotes around vendor value
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": malformed_json_string
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)

    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=True)
    assert result is None


def test_extract_invoice_data_malformed_json_content_gemini(mocker, mock_gemini_api_key):
    """Test handling of malformed JSON string within a successful API response (Gemini)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    malformed_json_string = "{\"invoice_no\": \"GEM-BAD\", \"total_amount\": 100.00 " # Missing closing brace
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": malformed_json_string}]
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)

    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=False)
    assert result is None

def test_extract_invoice_data_empty_image_bytes(mocker, mock_openai_api_key):
    """Test handling of empty image bytes input."""
    # No need to mock requests.post if the function returns early
    mock_post = mocker.patch('requests.post')
    result = extract_invoice_data(b"", use_openai=True) # Empty bytes
    assert result is None
    mock_post.assert_not_called() # API call should not be made

def test_extract_invoice_data_total_amount_conversion(mocker, mock_openai_api_key):
    """Test various total_amount string formats and their conversion to float."""
    test_cases = [
        ("123.45", 123.45),
        ("€1,234.56", 1234.56),
        ("£789", 789.0),
        ("$ 100.00", 100.00),
        ("5,000", 5000.0),
        ("InvalidAmount", "InvalidAmount"), # Should remain string if not convertible after cleaning
        (None, None),
        (123, 123.0) # Already a number
    ]

    for amount_str, expected_float in test_cases:
        mock_response = MagicMock()
        mock_response.status_code = 200
        api_content_json_string = json.dumps({
            "invoice_no": "AMT-TEST",
            "date": "2023-01-01",
            "vendor": "Amount Tester",
            "total_amount": amount_str
        })
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": api_content_json_string}}
            ]
        }
        
        # Reset mock for each iteration if needed, or use a fresh one
        # For this test, requests.post is patched within the loop to ensure fresh mock for each call
        with patch('requests.post', return_value=mock_response) as mock_post_call:
            result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=True)
            assert result is not None
            if isinstance(expected_float, float) or expected_float is None:
                assert result['total_amount'] == expected_float
            else: # If it was unparseable, it might be kept as original string or become None based on logic
                  # Current logic keeps the original string if it cannot be converted to float after cleaning.
                  # If it's a type error before cleaning (e.g. dict), it becomes None.
                assert result['total_amount'] == amount_str
            mock_post_call.assert_called_once()


def test_extract_invoice_data_no_content_in_response_openai(mocker, mock_openai_api_key):
    """Test OpenAI response that is valid JSON but missing the expected content field."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": { # Missing 'content' key here
                    "role": "assistant"
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)
    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=True)
    assert result is None

def test_extract_invoice_data_no_parts_in_gemini_response(mocker, mock_gemini_api_key):
    """Test Gemini response missing parts."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": { # Missing 'parts' key
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)
    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=False)
    assert result is None

def test_extract_invoice_data_gemini_part_text_empty(mocker, mock_gemini_api_key):
    """Test Gemini response where parts[0].text is empty."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": ""}] # Empty text
                }
            }
        ]
    }
    mocker.patch('requests.post', return_value=mock_response)
    result = extract_invoice_data(DUMMY_IMAGE_BYTES, use_openai=False)
    assert result is None

# To run these tests: `pytest tests/test_vision_extractor.py`
# Ensure you have an __init__.py in the tests folder if it's not automatically treated as a package.
# Also, ensure that the `src` directory is discoverable by pytest (e.g., by running pytest from the project root,
# or by setting PYTHONPATH, or by having a conftest.py that modifies sys.path).
# A common way is to have a pytest.ini or pyproject.toml that configures pythonpaths for pytest.
# Example pytest.ini:
# [pytest]
# python_paths = . src 