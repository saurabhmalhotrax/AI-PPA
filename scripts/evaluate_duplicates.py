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
    get_embeddings,
    build_faiss_index,
    find_potential_duplicates
)
from src.config import DUPLICATE_DETECTION

# --- Configuration ---
EXTRACTED_INVOICES_PATH = DUPLICATE_DETECTION.get('extracted_invoices_path', os.path.join(PROJECT_ROOT, 'data', 'extracted_invoices.csv'))
GROUND_TRUTH_PATH = DUPLICATE_DETECTION.get('ground_truth_path', os.path.join(PROJECT_ROOT, 'data', 'duplicate_pairs_ground_truth.csv'))
INVOICE_ID_COLUMN = DUPLICATE_DETECTION.get('invoice_id_column', 'filename')

# Default heuristic thresholds if not found in config (matching duplicate_detector.py)
DEFAULT_SIMILARITY_THRESHOLD_EVAL = 0.88
DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD_EVAL = 1.0
DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD_EVAL = 7
DEFAULT_TOP_K_FAISS_EVAL = 20

def load_data_for_evaluation():
    """Loads extracted invoices and ground truth data for evaluation."""
    if not os.path.exists(EXTRACTED_INVOICES_PATH):
        error_msg = f"Error: Extracted invoices file not found at {EXTRACTED_INVOICES_PATH}"
        print(error_msg)
        # For Streamlit, we might want to return None or raise error instead of sys.exit
        # For now, keeping print and sys.exit for standalone script, but Streamlit part will need to handle this
        raise FileNotFoundError(error_msg)
    if not os.path.exists(GROUND_TRUTH_PATH):
        error_msg = f"Error: Ground truth file not found at {GROUND_TRUTH_PATH}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
        
    all_invoices_df = pd.read_csv(EXTRACTED_INVOICES_PATH)
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    
    # Basic validation
    if INVOICE_ID_COLUMN not in all_invoices_df.columns:
        error_msg = f"Error: Invoice ID column '{INVOICE_ID_COLUMN}' not found in {EXTRACTED_INVOICES_PATH}. Available: {all_invoices_df.columns.tolist()}"
        print(error_msg)
        raise ValueError(error_msg)
    for col in ['invoice_id_1', 'invoice_id_2', 'is_duplicate']:
        if col not in ground_truth_df.columns:
            error_msg = f"Error: Column '{col}' not found in {GROUND_TRUTH_PATH}."
            print(error_msg)
            raise ValueError(error_msg)
            
    return all_invoices_df, ground_truth_df

def get_duplicate_detection_metrics():
    """
    Calculates and returns precision, recall, F1 score, and confusion matrix
    for the duplicate detection model.
    """
    all_invoices_df, ground_truth_df = load_data_for_evaluation()

    print(f"Loaded {len(all_invoices_df)} invoices for evaluation.")
    print(f"Loaded {len(ground_truth_df)} ground truth pairs for evaluation.")

    all_invoices_data_list = all_invoices_df.to_dict('records')

    print("Generating text representations for all invoices (evaluation)...")
    all_invoice_texts = [invoice_to_text_representation(invoice) for invoice in all_invoices_data_list]
    
    print("Generating embeddings for all invoices (evaluation)...")
    # Use batch get_embeddings
    invoice_embeddings_list = get_embeddings(all_invoice_texts, batch_size=32, show_progress_bar=True)
    print(f"Generated {len(invoice_embeddings_list)} embeddings for evaluation.")

    if not invoice_embeddings_list or not any(e is not None for e in invoice_embeddings_list):
        error_msg = "Error: No embeddings were generated for evaluation. Cannot build FAISS index."
        print(error_msg)
        raise ValueError(error_msg)

    # Filter out None embeddings and corresponding data if any (should not happen with current get_embeddings)
    valid_indices = [i for i, emb in enumerate(invoice_embeddings_list) if emb is not None]
    if len(valid_indices) != len(all_invoices_data_list):
        print("Warning: Some invoices could not be embedded. Filtering them out for evaluation.")
        all_invoices_data_list = [all_invoices_data_list[i] for i in valid_indices]
        invoice_embeddings_list = [invoice_embeddings_list[i] for i in valid_indices]
        if not invoice_embeddings_list:
            raise ValueError("Error: No valid embeddings left after filtering.")

    print("Building FAISS index (evaluation)...")
    faiss_index = build_faiss_index(invoice_embeddings_list)
    if faiss_index is None:
        error_msg = "Error: Failed to build FAISS index for evaluation."
        print(error_msg)
        raise RuntimeError(error_msg)
    print(f"FAISS index built with {faiss_index.ntotal} embeddings for evaluation.")

    embedding_id_map = {i: invoice_data[INVOICE_ID_COLUMN] for i, invoice_data in enumerate(all_invoices_data_list)}

    y_true = []
    y_pred = []
    
    # Get thresholds from config or use defaults
    similarity_threshold = DUPLICATE_DETECTION.get('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD_EVAL)
    amount_threshold = DUPLICATE_DETECTION.get('amount_difference_threshold', DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD_EVAL)
    date_threshold_days = DUPLICATE_DETECTION.get('date_difference_days_threshold', DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD_EVAL)
    top_k = DUPLICATE_DETECTION.get('top_k_faiss', DEFAULT_TOP_K_FAISS_EVAL)

    print("Evaluating duplicate pairs with current model settings...")
    for _, row in ground_truth_df.iterrows():
        invoice_id_1 = row['invoice_id_1']
        invoice_id_2 = row['invoice_id_2']
        is_duplicate_actual = int(row['is_duplicate'])
        
        if not any(inv[INVOICE_ID_COLUMN] == invoice_id_1 for inv in all_invoices_data_list):
            print(f"Warning: invoice_id_1 '{invoice_id_1}' from ground truth not found in extracted invoices. Skipping pair.")
            continue

        # Find the index of invoice_id_1 in all_invoices_data_list to get its pre-computed embedding
        # This is safer than relying on find_potential_duplicates to re-embed if it were designed that way
        # However, find_potential_duplicates expects the *full* list of embeddings.
        
        potential_dupes = find_potential_duplicates(
            target_invoice_id=invoice_id_1,
            all_invoices_data=all_invoices_data_list,
            invoice_embeddings=invoice_embeddings_list,
            faiss_index=faiss_index,
            embedding_id_map=embedding_id_map,
            top_k=top_k, 
            similarity_threshold=similarity_threshold,
            amount_threshold=amount_threshold,
            date_threshold_days=date_threshold_days
        )

        predicted_as_duplicate = 0
        for dupe in potential_dupes:
            if dupe['matched_invoice_id'] == invoice_id_2:
                predicted_as_duplicate = 1
                break
        
        y_true.append(is_duplicate_actual)
        y_pred.append(predicted_as_duplicate)

    if not y_true:
        error_msg = "Error: No pairs were processed from the ground truth for evaluation. Check ground truth file and invoice IDs."
        print(error_msg)
        # Consider what to return or raise if Streamlit calls this
        return 0.0, 0.0, 0.0, pd.DataFrame()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # For better display in Streamlit, convert confusion matrix to DataFrame
    # Ensure labels are correct if not just 0 and 1
    # Assuming 0 = Not Duplicate, 1 = Duplicate
    cm_labels = ['Not Duplicate', 'Duplicate']
    if len(cm) == 2:
        confusion_matrix_df = pd.DataFrame(cm, 
                                           index=[f'Actual: {cm_labels[0]}', f'Actual: {cm_labels[1]}'], 
                                           columns=[f'Predicted: {cm_labels[0]}', f'Predicted: {cm_labels[1]}'])
    else:
        print(f"Warning: Confusion matrix is not 2x2: {cm}. Displaying raw.")
        confusion_matrix_df = pd.DataFrame(cm)

    print("\n--- Evaluation Metrics (Calculated) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix_df)
    
    return precision, recall, f1, confusion_matrix_df

if __name__ == '__main__':
    print("Running Duplicate Detection Model Evaluation...")
    try:
        precision, recall, f1, cm_df = get_duplicate_detection_metrics()
        # The function already prints, but we can add a summary here if needed for script execution
        print("\nEvaluation Complete.")
        print(f"Summary: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during evaluation: {e}")
        print(traceback.format_exc())
        sys.exit(1) 