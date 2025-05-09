import os
from dotenv import load_dotenv  # type: ignore

# Load environment variables from .env file
# Attempting to force a re-read by adding a comment here
load_dotenv()

# Vision AI API Keys - these should be set in your .env file
# Used by src/vision_extractor.py and scripts/batch_extract.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Other Configuration ---

# Endpoints and Model Names (can be overridden by .env variables if needed, or kept as defaults)
OPENAI_VISION_ENDPOINT = os.getenv("OPENAI_VISION_ENDPOINT", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

GEMINI_VISION_ENDPOINT = os.getenv("GEMINI_VISION_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent") # Example
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-vision-latest") # Example

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Generic VISION_API_KEY - assess if still needed or if specific keys above cover all uses.
# If this was intended for the vision_extractor, it now uses OPENAI_API_KEY or GEMINI_API_KEY.
VISION_API_KEY = os.getenv("VISION_API_KEY")

# Basic check to guide user if keys are missing
if not OPENAI_API_KEY and not GEMINI_API_KEY:
    print("Warning: Neither OPENAI_API_KEY nor GEMINI_API_KEY are found in environment variables.")
    print("Please set them in your .env file if you plan to use vision extraction features.")
elif not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables. OpenAI features will not work.")
elif not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables. Gemini features will not work.")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    print("Warning: One or more Neo4j connection variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) are not set in environment variables.")
    print("Please set them in your .env file for Neo4j integration to work.")

# Add other configurations as needed
# Example: EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

DUPLICATE_DETECTION = {
    'extracted_invoices_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'extracted_invoices.csv'),
    'ground_truth_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'duplicate_pairs_ground_truth.csv'),
    'invoice_id_column': 'filename',  # Or 'id', ensure this matches your CSVs and ground truth
    'similarity_threshold': 0.88,    # Default from evaluate_duplicates.py & duplicate_detector.py
    'amount_difference_threshold': 1.0, # Default
    'date_difference_days_threshold': 7, # Default
    'top_k_faiss': 20                 # Default
}

# You might want to use a single key if your chosen Vision AI provider is fixed
# For example, if you only plan to use OpenAI:
# VISION_API_KEY = OPENAI_API_KEY
# Or if you decide dynamically:
# VISION_API_KEY = None # To be set at runtime or based on selection 