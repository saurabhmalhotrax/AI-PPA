import base64
import json
import logging
import requests
from typing import Union, Optional

# Attempt to import API keys and endpoints from config
# If this script is run directly and src.config is not found,
# it will fall back to environment variables or placeholders.
try:
    from src.config import (
        OPENAI_API_KEY,
        OPENAI_VISION_ENDPOINT,
        OPENAI_MODEL,
        GEMINI_API_KEY,
        GEMINI_VISION_ENDPOINT,
        GEMINI_MODEL
    )
except ImportError:
    # Fallback or placeholder if run outside a project structure or config is missing
    # In a real application, ensure your deployment strategy handles API keys securely (e.g., env variables)
    import os
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_FALLBACK")
    OPENAI_VISION_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    OPENAI_MODEL = "gpt-4o"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_FALLBACK")
    GEMINI_VISION_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent" # Example
    GEMINI_MODEL = "gemini-1.5-pro-vision-latest" # Example

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_invoice_data(image_bytes: bytes, use_openai: bool = True) -> Optional[dict]:
    """
    Extracts structured data from an invoice image using a Vision AI API.

    Args:
        image_bytes: The byte content of the image file.
        use_openai: If True, uses OpenAI GPT-4o Vision. Otherwise, uses Google Gemini 1.5 Pro Vision.

    Returns:
        A dictionary containing the extracted fields:
        {"invoice_no": str | None, "date": str | None, "vendor": str | None, "total_amount": float | None}
        Returns None if extraction fails or an error occurs.
    """
    if not image_bytes:
        logging.error("Image bytes are empty.")
        return None

    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    headers = {}
    payload = {}
    api_key = ""
    endpoint = ""
    
    common_prompt_text = ("Extract the following fields from this invoice image and return them as a valid JSON object: "
                          "invoice_no (invoice number or ID), "
                          "date (invoice date in YYYY-MM-DD format if possible, otherwise as seen), "
                          "vendor (supplier or vendor name), "
                          "total_amount (the final total amount due as a number, convert if necessary). "
                          "If a field is not found, use null for its value. "
                          "The output MUST be a single, minified JSON object with no surrounding text or markdown.")

    if use_openai:
        api_key = OPENAI_API_KEY
        endpoint = OPENAI_VISION_ENDPOINT
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": common_prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}" 
                                # Assuming JPEG, could be dynamic based on actual image type if known
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500, # Increased slightly to ensure JSON fits
            "temperature": 0.1 # Low temperature for deterministic JSON output
        }
    else: # Use Gemini
        api_key = GEMINI_API_KEY
        endpoint = f"{GEMINI_VISION_ENDPOINT}?key={api_key}" # Gemini API key is often part of the URL
        headers = {
            "Content-Type": "application/json"
        }
        # Gemini's payload structure is different. This is a common pattern:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": common_prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg", # Assuming JPEG
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 500, # Ensure this is sufficient
                # "responseMimeType": "application/json" # Some Gemini versions support this
            }
        }
        # Note: For Gemini, if "responseMimeType": "application/json" is supported by the specific model endpoint,
        # it can simplify response parsing. Otherwise, the model needs to be prompted explicitly for JSON.

    try:
        logging.info(f"Sending request to {'OpenAI' if use_openai else 'Gemini'} API at {endpoint.split('?')[0]}")
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60) # 60-second timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        response_data = response.json()
        extracted_json_str = None

        if use_openai:
            # OpenAI typically returns the JSON string in choices[0].message.content
            # Sometimes it might be wrapped in markdown ```json ... ```
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                # Try to find JSON block if markdown is present
                if content.strip().startswith("```json"):
                    extracted_json_str = content.strip()[7:-3].strip()
                elif content.strip().startswith("{") and content.strip().endswith("}"):
                    extracted_json_str = content.strip()
                else: # Fallback for unexpected wrapping or direct JSON string
                    extracted_json_str = content 
            else:
                logging.error("OpenAI API response does not contain expected content structure.")
                return None
        else: # Gemini
            # Gemini response structure can vary. This is a common pattern.
            # It might be in parts[0].text or similar, potentially wrapped.
            # If "responseMimeType": "application/json" was used and worked, this might be simpler.
            candidates = response_data.get("candidates", [])
            if candidates and candidates[0].get("content", {}).get("parts", []):
                raw_text = candidates[0]["content"]["parts"][0].get("text", "")
                if raw_text:
                    if raw_text.strip().startswith("```json"):
                        extracted_json_str = raw_text.strip()[7:-3].strip()
                    elif raw_text.strip().startswith("{") and raw_text.strip().endswith("}"):
                        extracted_json_str = raw_text.strip()
                    else: # Fallback
                         extracted_json_str = raw_text
                else:
                    logging.error("Gemini API response part text is empty.")
                    return None
            else:
                logging.error("Gemini API response does not contain expected candidate structure.")
                return None
        
        if not extracted_json_str:
            logging.error(f"Could not extract JSON string from API response. Raw response: {response_data}")
            return None

        logging.info(f"Extracted JSON string: {extracted_json_str}")
        
        # Parse the extracted JSON string
        try:
            # The model might sometimes return a JSON string that needs to be parsed again
            if isinstance(extracted_json_str, str):
                 # Clean potential escape characters if the string itself is an escaped JSON
                try:
                    # First, try to parse directly if it's a clean JSON string
                    data = json.loads(extracted_json_str)
                except json.JSONDecodeError:
                    # If direct parsing fails, it might be an escaped JSON string within a string
                    # e.g. ""{\"invoice_no\": \"123\"...}""
                    # This attempts to unescape and parse
                    logging.warning(f"Direct JSON parsing failed, trying to unescape and parse: {extracted_json_str}")
                    data = json.loads(json.loads(f'"{extracted_json_str}"'))

            elif isinstance(extracted_json_str, dict): # Already a dict
                data = extracted_json_str
            else:
                logging.error(f"Unexpected type for extracted_json_str: {type(extracted_json_str)}")
                return None

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse extracted JSON string: {extracted_json_str}. Error: {e}")
            # Log a snippet of the problematic string for easier debugging
            logging.error(f"Problematic JSON snippet (first 200 chars): {extracted_json_str[:200]}")
            return None
        
        # Validate and structure the output
        invoice_details = {
            "invoice_no": data.get("invoice_no"),
            "date": data.get("date"),
            "vendor": data.get("vendor"),
            "total_amount": data.get("total_amount")
        }

        # Optional: Convert total_amount to float
        if invoice_details["total_amount"] is not None:
            try:
                # Remove currency symbols, commas, and then convert
                amount_str = str(invoice_details["total_amount"]).strip()
                # More robust cleaning: remove common currency symbols and thousand separators
                for char_to_remove in ['$', '€', '£', ',', ' ']: # Add more symbols if needed
                    amount_str = amount_str.replace(char_to_remove, '')
                
                invoice_details["total_amount"] = float(amount_str)
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not convert total_amount '{invoice_details['total_amount']}' to float. Error: {e}. Keeping as is or setting to None.")
                # Decide on fallback: keep original string or set to None
                # For now, let's keep the original problematic string if conversion fails, or set to None if it was already None
                if not isinstance(invoice_details["total_amount"], (str, int, float)):
                     invoice_details["total_amount"] = None


        logging.info(f"Successfully extracted invoice data: {invoice_details}")
        return invoice_details

    except requests.exceptions.RequestException as e:
        logging.error(f"Network or HTTP error calling Vision API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"API Response Status: {e.response.status_code}")
            logging.error(f"API Response Body: {e.response.text}")
        return None
    except json.JSONDecodeError as e: # For issues with response.json() itself
        logging.error(f"Failed to parse main API JSON response: {e}")
        if 'response' in locals() and response is not None:
             logging.error(f"Raw API response text (first 200 chars): {response.text[:200]}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example usage (requires a sample image file)
    # Create a dummy image file for testing if you don't have one
    # This is a very basic example; real testing should use actual invoice images.
    
    logging.info("Starting example vision extraction.")
    
    # Ensure OPENAI_API_KEY (and GEMINI_API_KEY if testing Gemini) are set in src/config.py or as environment variables
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_FALLBACK" or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        logging.warning("OpenAI API key is not set. Please set it in src/config.py or as an environment variable.")
        # You could exit here or skip the OpenAI part of the test
    
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_FALLBACK" or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logging.warning("Gemini API key is not set. Please set it in src/config.py or as an environment variable.")

    # Create a tiny dummy PNG image bytes for basic API call testing (won't extract meaningful data)
    # (1x1 transparent PNG)
    dummy_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    dummy_image_bytes = base64.b64decode(dummy_image_b64)

    # Test with OpenAI (if key is available)
    if not (OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_FALLBACK" or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE"):
        logging.info("\n--- Testing with OpenAI --- (using a dummy 1x1 pixel image)")
        extracted_data_openai = extract_invoice_data(dummy_image_bytes, use_openai=True)
        if extracted_data_openai:
            print("OpenAI - Extracted data:", json.dumps(extracted_data_openai, indent=2))
        else:
            print("OpenAI - Failed to extract data.")
    else:
        print("\n--- Skipping OpenAI test: API key not configured ---")

    # Test with Gemini (if key is available)
    # Note: Gemini Vision API might have stricter input requirements or different free tier limitations
    # This dummy image might not be sufficient for a successful Gemini call.
    if not (GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_FALLBACK" or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE"):
        logging.info("\n--- Testing with Gemini --- (using a dummy 1x1 pixel image)")
        extracted_data_gemini = extract_invoice_data(dummy_image_bytes, use_openai=False)
        if extracted_data_gemini:
            print("Gemini - Extracted data:", json.dumps(extracted_data_gemini, indent=2))
        else:
            print("Gemini - Failed to extract data.")
    else:
        print("\n--- Skipping Gemini test: API key not configured ---")

    # To test with a real image:
    # try:
    #     with open("path/to/your/invoice_image.jpg", "rb") as image_file:
    #         image_bytes_real = image_file.read()
    #     
    #     logging.info("\n--- Testing with REAL IMAGE using OpenAI ---")
    #     data_real_openai = extract_invoice_data(image_bytes_real, use_openai=True)
    #     if data_real_openai:
    #         print("OpenAI (Real Image) - Extracted data:", json.dumps(data_real_openai, indent=2))
    #     else:
    #         print("OpenAI (Real Image) - Failed to extract data.")

    # except FileNotFoundError:
    #     logging.warning("Real image file not found. Skipping real image test.")
    # except Exception as e:
    #     logging.error(f"Error during real image test: {e}") 