import os
import logging
from PIL import Image
from datasets import load_dataset, DownloadConfig # type: ignore 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DEFAULT_NUM_SAMPLES = 200
OUTPUT_DIR = "data/sample_invoices"
DATASET_NAME = "aharley/rvl_cdip"
INVOICE_LABEL = 11  # Label for invoices in RVL-CDIP

def create_output_directory(directory: str):
    """Creates the output directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Ensured output directory exists: {directory}")
    except OSError as e:
        logging.error(f"Error creating directory {directory}: {e}")
        raise

def download_and_save_invoices(num_samples: int = DEFAULT_NUM_SAMPLES, output_dir: str = OUTPUT_DIR):
    """Loads the RVL-CDIP dataset, filters for invoices, and saves a specified number of samples."""
    create_output_directory(output_dir)

    logging.info(f"Attempting to load dataset: {DATASET_NAME} with streaming.")
    
    # Using streaming=True to avoid downloading the entire dataset at once.
    # Increased timeout for dataset loading, as rvl-cdip can be slow to initialize connection
    # This might require `pip install fsspec aiohttp` if not already installed by `datasets`
    # Set trust_remote_code=True as per Hugging Face Hub requirements for this dataset.
    try:
        dataset = load_dataset(DATASET_NAME, split='train', streaming=True, trust_remote_code=True)
    except Exception as e:
        logging.error(f"Failed to load dataset {DATASET_NAME}: {e}")
        logging.error("Please ensure you have a stable internet connection and the Hugging Face Hub is accessible.")
        logging.error("You might need to install additional dependencies like `fsspec` and `aiohttp`: pip install fsspec aiohttp")
        return

    logging.info(f"Successfully started streaming dataset. Filtering for invoices (label {INVOICE_LABEL}).")

    saved_count = 0
    processed_count = 0
    max_processed_to_find_samples = 50000 # Limit how many items we check to find enough invoices

    try:
        for i, example in enumerate(dataset):
            processed_count += 1
            if saved_count >= num_samples:
                logging.info(f"Collected {saved_count} invoice samples. Stopping.")
                break

            if example.get('label') == INVOICE_LABEL:
                image = example.get('image')
                if image and isinstance(image, Image.Image):
                    # Generate a unique filename. Using a simple counter.
                    # You could use example.get('id') or some other unique identifier if available and preferred.
                    filename = f"rvl_cdip_invoice_{saved_count + 1}.png" 
                    filepath = os.path.join(output_dir, filename)

                    if os.path.exists(filepath):
                        logging.info(f"Image {filepath} already exists. Skipping.")
                        # We could increment saved_count here if we want to count existing files towards the total
                        # For now, we only count newly saved files or files we attempt to save.
                        # To strictly get N *new* files, this logic might need adjustment or we ensure the dir is empty.
                        # Let's assume for now we just want to ensure N files are present, and skip if already there.
                        saved_count +=1 # Count it as one of the samples obtained
                        continue

                    try:
                        image.save(filepath, "PNG")
                        logging.info(f"Saved image {saved_count + 1}/{num_samples}: {filepath}")
                        saved_count += 1
                    except IOError as e:
                        logging.error(f"Error saving image {filepath}: {e}")
                    except Exception as e:
                        logging.error(f"An unexpected error occurred while saving image {filepath}: {e}")
                else:
                    logging.warning(f"Sample {i} has label {INVOICE_LABEL} but no valid image data.")
            
            if processed_count % 1000 == 0:
                logging.info(f"Processed {processed_count} dataset entries... Found {saved_count} invoices so far.")

            if processed_count >= max_processed_to_find_samples and saved_count < num_samples:
                logging.warning(f"Processed {max_processed_to_find_samples} entries but only found {saved_count}/{num_samples} invoices. Stopping.")
                break

    except Exception as e:
        logging.error(f"An error occurred while iterating through the dataset: {e}")
        logging.error("This could be due to network issues or problems with the dataset stream.")

    if saved_count < num_samples:
        logging.warning(f"Could only save {saved_count} out of {num_samples} requested invoice samples.")
    else:
        logging.info(f"Successfully saved {saved_count} invoice images to {output_dir}")

if __name__ == "__main__":
    # Allow configuring the number of samples via command-line argument in a real script
    # For this implementation, we use the default.
    num_to_download = DEFAULT_NUM_SAMPLES 
    # Example: if you wanted to pass an argument:
    # import argparse
    # parser = argparse.ArgumentParser(description="Download sample invoice images from RVL-CDIP.")
    # parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of invoice samples to download.")
    # args = parser.parse_args()
    # num_to_download = args.num_samples

    logging.info(f"Starting script to download {num_to_download} invoice samples.")
    download_and_save_invoices(num_samples=num_to_download)
    logging.info("Script finished.") 