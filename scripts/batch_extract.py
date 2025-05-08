import os
import pandas as pd # type: ignore
import logging
import time

# Attempt to import the extraction function and config details
try:
    from src.vision_extractor import extract_invoice_data
    # Import specific keys/models from config to check if they are set,
    # as vision_extractor.py will use these.
    from src.config import OPENAI_API_KEY, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL
except ImportError as e:
    logging.error(f"Error importing necessary modules: {e}. Ensure src/vision_extractor.py and src/config.py exist and are in PYTHONPATH.")
    # Optionally, exit if critical modules are missing
    # exit(1) 
    # For now, define placeholders so the rest of the script can be parsed,
    # but it will likely fail at runtime if imports truly failed.
    def extract_invoice_data(*args, **kwargs): # type: ignore
        logging.error("extract_invoice_data function is not available due to import error.")
        return None
    OPENAI_API_KEY = "PLACEHOLDER_DUE_TO_IMPORT_ERROR"
    OPENAI_MODEL = "PLACEHOLDER_DUE_TO_IMPORT_ERROR"
    GEMINI_API_KEY = "PLACEHOLDER_DUE_TO_IMPORT_ERROR"
    GEMINI_MODEL = "PLACEHOLDER_DUE_TO_IMPORT_ERROR"


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_IMAGE_DIR = "data/sample_invoices/"
OUTPUT_CSV_FILE = "data/extracted_invoices.csv"
USE_OPENAI_FOR_BATCH = True  # Set to False to use Gemini
# Add a small delay between API calls to avoid hitting rate limits, if necessary
SLEEP_BETWEEN_CALLS_SECONDS = 1 # Adjust as needed, 0 for no delay

# Define expected columns for the CSV file to ensure consistency
CSV_COLUMNS = ['filename', 'invoice_no', 'date', 'vendor', 'total_amount', 'error']

def check_api_keys_configured():
    """Checks if the relevant API key (OpenAI or Gemini) is configured and not a placeholder."""
    key_to_check = None
    key_name = ""
    model_name_val = ""
    
    if USE_OPENAI_FOR_BATCH:
        key_to_check = OPENAI_API_KEY
        key_name = "OPENAI_API_KEY"
        model_name_val = OPENAI_MODEL
        logging.info(f"Using OpenAI for batch extraction with model: {model_name_val}")
    else:
        key_to_check = GEMINI_API_KEY
        key_name = "GEMINI_API_KEY"
        model_name_val = GEMINI_MODEL
        logging.info(f"Using Gemini for batch extraction with model: {model_name_val}")

    if key_to_check is None or not key_to_check: # Check if None or empty string
        logging.error(f"{key_name} is not set in your .env file or is empty.")
        logging.error("Please set your API key to proceed with batch extraction.")
        return False
    logging.info(f"{key_name} appears to be configured.")
    return True

def process_images_in_directory(image_dir: str, output_csv: str):
    """
    Processes all images in a directory, extracts invoice data, and saves to a CSV.
    """
    if not os.path.exists(image_dir):
        logging.error(f"Input image directory not found: {image_dir}")
        return

    if not check_api_keys_configured():
        logging.warning("API keys not configured. Aborting batch extraction.")
        return

    all_extracted_data = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        logging.info(f"No image files found in {image_dir}")
        return

    logging.info(f"Found {len(image_files)} images to process in {image_dir}.")

    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        logging.info(f"Processing image {i+1}/{len(image_files)}: {filename}")
        
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            extracted_info = extract_invoice_data(image_bytes, use_openai=USE_OPENAI_FOR_BATCH)
            
            if extracted_info:
                # Ensure all expected keys are present, defaulting to None if not
                row_data = {'filename': filename}
                for col in CSV_COLUMNS:
                    if col != 'filename' and col != 'error':
                        row_data[col] = extracted_info.get(col)
                row_data['error'] = None # No error if we got here and extracted_info is not None
                all_extracted_data.append(row_data)
                logging.info(f"Successfully extracted data for {filename}.")
            else:
                logging.warning(f"Failed to extract data for {filename} (extractor returned None).")
                # Add a row indicating failure for this file
                row_data_failure = {'filename': filename}
                for col in CSV_COLUMNS:
                     if col != 'filename' and col != 'error':
                        row_data_failure[col] = None
                row_data_failure['error'] = "ExtractionFailedOrNoData"
                all_extracted_data.append(row_data_failure)

        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}. Skipping.")
            all_extracted_data.append({col: (filename if col == 'filename' else ("FileNotFound" if col == 'error' else None)) for col in CSV_COLUMNS})
        except IOError as e:
            logging.error(f"IOError reading image {image_path}: {e}. Skipping.")
            all_extracted_data.append({col: (filename if col == 'filename' else (f"IOError: {e}" if col == 'error' else None)) for col in CSV_COLUMNS})
        except Exception as e:
            logging.error(f"An unexpected error occurred processing {filename}: {e}", exc_info=True)
            all_extracted_data.append({col: (filename if col == 'filename' else (f"UnexpectedError: {e}" if col == 'error' else None)) for col in CSV_COLUMNS})
        
        if SLEEP_BETWEEN_CALLS_SECONDS > 0 and i < len(image_files) - 1 :
            logging.info(f"Sleeping for {SLEEP_BETWEEN_CALLS_SECONDS}s...")
            time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)


    if all_extracted_data:
        # Create directory for CSV if it doesn't exist
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created directory for output CSV: {output_dir}")

        df = pd.DataFrame(all_extracted_data)
        
        # Reorder columns to match CSV_COLUMNS and ensure all are present
        df = df.reindex(columns=CSV_COLUMNS)

        try:
            df.to_csv(output_csv, index=False)
            logging.info(f"Successfully saved extracted data for {len(df[df['error'].isnull()])} images to {output_csv}")
            num_errors = len(df[df['error'].notnull()])
            if num_errors > 0:
                logging.warning(f"{num_errors} images encountered errors during processing. See CSV for details.")
        except Exception as e:
            logging.error(f"Failed to save data to CSV {output_csv}: {e}")
    else:
        logging.info("No data was extracted from any images.")

if __name__ == "__main__":
    logging.info("Starting batch invoice extraction script.")
    
    # Check if core components were imported and API keys are available before proceeding
    # The extract_invoice_data function itself will use the keys from src.config
    if 'extract_invoice_data' not in globals() or not callable(extract_invoice_data) or OPENAI_API_KEY == "PLACEHOLDER_DUE_TO_IMPORT_ERROR":
        logging.critical("Core components (vision_extractor or config) could not be imported correctly. Aborting.")
    else:
        process_images_in_directory(INPUT_IMAGE_DIR, OUTPUT_CSV_FILE)
    
    logging.info("Batch extraction script finished.") 