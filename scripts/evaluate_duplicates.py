import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

# Adjust path to import from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.duplicate_detector import (
    invoice_to_text_representation,
    get_embedding,
    build_faiss_index,
    find_potential_duplicates
)

# --- Configuration ---
EXTRACTED_INVOICES_PATH = os.path.join(PROJECT_ROOT, 'data', 'extracted_invoices.csv')
GROUND_TRUTH_PATH = os.path.join(PROJECT_ROOT, 'data', 'duplicate_pairs_ground_truth.csv')
# Ensure this ID column matches the one in your CSVs and used in ground truth
INVOICE_ID_COLUMN = 'filename' # Or 'id' if that's what you use

def load_data():
    """Loads extracted invoices and ground truth data."""
    if not os.path.exists(EXTRACTED_INVOICES_PATH):
        print(f"Error: Extracted invoices file not found at {EXTRACTED_INVOICES_PATH}")
        print("Please ensure you have run the extraction process and the file exists.")
        sys.exit(1)
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Error: Ground truth file not found at {GROUND_TRUTH_PATH}")
        print("Please ensure you have created and populated this file.")
        sys.exit(1)
        
    all_invoices_df = pd.read_csv(EXTRACTED_INVOICES_PATH)
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    
    # Basic validation
    if INVOICE_ID_COLUMN not in all_invoices_df.columns:
        print(f"Error: Invoice ID column '{INVOICE_ID_COLUMN}' not found in {EXTRACTED_INVOICES_PATH}.")
        print(f"Available columns: {all_invoices_df.columns.tolist()}")
        sys.exit(1)
    for col in ['invoice_id_1', 'invoice_id_2', 'is_duplicate']:
        if col not in ground_truth_df.columns:
            print(f"Error: Column '{col}' not found in {GROUND_TRUTH_PATH}.")
            sys.exit(1)
            
    return all_invoices_df, ground_truth_df

def main():
    """Main evaluation logic."""
    all_invoices_df, ground_truth_df = load_data()

    print(f"Loaded {len(all_invoices_df)} invoices from {EXTRACTED_INVOICES_PATH}")
    print(f"Loaded {len(ground_truth_df)} ground truth pairs from {GROUND_TRUTH_PATH}")

    # Convert DataFrame to list of dicts for processing
    all_invoices_data_list = all_invoices_df.to_dict('records')

    print("Generating text representations for all invoices...")
    all_invoice_texts = [invoice_to_text_representation(invoice) for invoice in all_invoices_data_list]
    
    print("Generating embeddings for all invoices...")
    # This can take time for many invoices
    invoice_embeddings_list = [get_embedding(text) for text in all_invoice_texts]
    print(f"Generated {len(invoice_embeddings_list)} embeddings.")

    if not invoice_embeddings_list or not any(invoice_embeddings_list):
        print("Error: No embeddings were generated. Cannot build FAISS index.")
        sys.exit(1)

    print("Building FAISS index...")
    faiss_index = build_faiss_index(invoice_embeddings_list)
    if faiss_index is None:
        print("Error: Failed to build FAISS index.")
        sys.exit(1)
    print(f"FAISS index built with {faiss_index.ntotal} embeddings.")

    # Create a mapping from FAISS index (0 to N-1) to the original invoice ID
    # This assumes the order of all_invoices_data_list is preserved in the embeddings and FAISS index
    embedding_id_map = {i: invoice_data[INVOICE_ID_COLUMN] for i, invoice_data in enumerate(all_invoices_data_list)}

    y_true = []
    y_pred = []
    false_negatives_list = [] # Initialize list for false negatives

    print("Evaluating duplicate pairs...")
    for _, row in ground_truth_df.iterrows():
        invoice_id_1 = row['invoice_id_1']
        invoice_id_2 = row['invoice_id_2']
        is_duplicate_actual = int(row['is_duplicate'])
        
        # Ensure invoice_id_1 exists in our loaded invoices
        if not any(inv[INVOICE_ID_COLUMN] == invoice_id_1 for inv in all_invoices_data_list):
            print(f"Warning: invoice_id_1 '{invoice_id_1}' from ground truth not found in extracted invoices. Skipping pair.")
            continue

        # Use the find_potential_duplicates function
        # The function's parameters (top_k, thresholds) can be adjusted here if needed for tuning
        potential_dupes = find_potential_duplicates(
            target_invoice_id=invoice_id_1,
            all_invoices_data=all_invoices_data_list,
            invoice_embeddings=invoice_embeddings_list,
            faiss_index=faiss_index,
            embedding_id_map=embedding_id_map
            # top_k=5, # Default is 5 in function
            # similarity_threshold=0.90, # Default is 0.90
            # amount_threshold=1.0, # Default is 1.0
            # date_threshold_days=7 # Default is 7
        )

        predicted_as_duplicate = 0
        for dupe in potential_dupes:
            if dupe['matched_invoice_id'] == invoice_id_2:
                predicted_as_duplicate = 1
                break
        
        y_true.append(is_duplicate_actual)
        y_pred.append(predicted_as_duplicate)

        if is_duplicate_actual == 1 and predicted_as_duplicate == 0:
            false_negatives_list.append((invoice_id_1, invoice_id_2))

    if not y_true:
        print("Error: No pairs were processed from the ground truth. Check ground truth file and invoice IDs.")
        sys.exit(1)

    print("\n--- Evaluation Metrics ---")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive']))
    
    if false_negatives_list:
        print("\n--- False Negatives (Missed Duplicates) ---")
        for fn_pair in false_negatives_list:
            print(f"Pair: {fn_pair[0]} --- {fn_pair[1]}")

    print("\nNote: Ensure your ground truth data is comprehensive and accurately labeled.")
    print(f"Target F1-score is >= 0.85. Current F1-score: {f1:.4f}")

if __name__ == "__main__":
    main() 