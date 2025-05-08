import argparse
import json
import pandas as pd
import sys
import os

# Adjust path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.duplicate_detector import (
    load_all_invoices_data,
    initialize_embedding_model,
    build_or_load_faiss_index,
    find_potential_duplicates,
    INVOICE_ID_COLUMN,
    TEXT_REPRESENTATION_COLUMN,
    EMBEDDING_COLUMN,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD,
    DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD,
    DEFAULT_TOP_K_FAISS
)

# Define paths (adjust if your project structure is different)
EXTRACTED_INVOICES_PATH = "data/extracted_invoices.csv"
FAISS_INDEX_PATH = "models/faiss_index.idx"
FAISS_IDS_PATH = "models/faiss_ids.pkl"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def main():
    parser = argparse.ArgumentParser(description="Find duplicate invoices for a given invoice ID.")
    parser.add_argument("invoice_id", type=str, help="The ID of the invoice to find duplicates for.")
    args = parser.parse_args()

    target_invoice_id = args.invoice_id

    try:
        print(f"Starting duplicate search for: {target_invoice_id}", file=sys.stderr)

        # 1. Load all invoice data
        print("Loading invoice data...", file=sys.stderr)
        all_invoices_df = load_all_invoices_data(EXTRACTED_INVOICES_PATH)
        if all_invoices_df.empty:
            print(json.dumps({"error": "Failed to load invoice data or data is empty."}))
            return
        print(f"Loaded {len(all_invoices_df)} invoices.", file=sys.stderr)

        if target_invoice_id not in all_invoices_df[INVOICE_ID_COLUMN].values:
            print(json.dumps({"error": f"Target invoice ID '{target_invoice_id}' not found in the dataset."}))
            return

        # 2. Initialize embedding model
        print("Initializing embedding model...", file=sys.stderr)
        embedding_model = initialize_embedding_model(EMBEDDING_MODEL_NAME)
        print("Embedding model initialized.", file=sys.stderr)

        # 3. Build or load FAISS index
        # Ensure the text representation and embeddings are ready for FAISS
        if TEXT_REPRESENTATION_COLUMN not in all_invoices_df.columns:
             print(json.dumps({"error": f"'{TEXT_REPRESENTATION_COLUMN}' not found in invoices. Preprocessing might be missing."}))
             return
        
        # Ensure embeddings are generated if not present (though find_potential_duplicates also handles this)
        if EMBEDDING_COLUMN not in all_invoices_df.columns:
            print(f"'{EMBEDDING_COLUMN}' not found, generating embeddings. This might take a moment.", file=sys.stderr)
            all_invoices_df[EMBEDDING_COLUMN] = all_invoices_df[TEXT_REPRESENTATION_COLUMN].apply(lambda x: embedding_model.encode(x))
            print("Embeddings generated.", file=sys.stderr)
        
        print("Building/loading FAISS index...", file=sys.stderr)
        faiss_index, faiss_invoice_ids = build_or_load_faiss_index(
            all_invoices_df,
            embedding_model,
            rebuild_index=False, # Set to True if you always want to rebuild
            index_path=FAISS_INDEX_PATH,
            ids_path=FAISS_IDS_PATH
        )
        if faiss_index is None:
            print(json.dumps({"error": "Failed to build or load FAISS index."}))
            return
        print("FAISS index ready.", file=sys.stderr)
        
        # 4. Find potential duplicates
        print(f"Finding potential duplicates for {target_invoice_id}...", file=sys.stderr)
        potential_duplicates = find_potential_duplicates(
            target_invoice_id=target_invoice_id,
            all_invoices_df=all_invoices_df,
            embedding_model=embedding_model, # Pass the model itself
            faiss_index=faiss_index,
            faiss_invoice_ids=faiss_invoice_ids, # Pass the loaded/built FAISS IDs
            similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
            amount_threshold=DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD,
            date_threshold=DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD,
            top_k=DEFAULT_TOP_K_FAISS
        )
        print("Duplicate search complete.", file=sys.stderr)

        # 5. Print results as JSON to stdout
        print(json.dumps(potential_duplicates, indent=4))

    except Exception as e:
        print(json.dumps({"error": f"An error occurred: {str(e)}"}), file=sys.stderr)
        # Also print to stdout for the parent process to capture as a primary error message if needed
        print(json.dumps({"error": f"An error occurred: {str(e)}"}))


if __name__ == "__main__":
    main() 