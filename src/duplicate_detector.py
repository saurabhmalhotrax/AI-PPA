from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
from typing import Union, Tuple, List
from dateutil import parser

# Initialize the embedding model globally - REMOVED
# model = SentenceTransformer('all-MiniLM-L6-v2') 

# Global variable to cache the model
_model_cache = None

def _get_embedding_model():
    """Loads and returns the SentenceTransformer model, caching it globally."""
    global _model_cache
    if _model_cache is None:
        print("INFO: Loading sentence_transformers model for the first time (forcing CPU).")
        # Force CPU to see if it resolves Metal/MPS related crashes
        _model_cache = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') 
        print("INFO: Sentence_transformers model loaded on CPU.")
    return _model_cache

def invoice_to_text_representation(invoice_data_dict: dict) -> str:
    """
    Converts a dictionary of structured invoice data into a single textual representation.

    Args:
        invoice_data_dict: A dictionary containing extracted invoice fields.
                           Expected keys include 'vendor', 'invoice_no', 'date', 'total_amount'.

    Returns:
        A single string representation of the invoice.
    """
    # Concatenate key fields into a single descriptive string.
    # Handle missing fields gracefully by using .get() with a default empty string.
    vendor = invoice_data_dict.get('vendor', '')
    invoice_no = invoice_data_dict.get('invoice_no', '')
    date = invoice_data_dict.get('date', '')
    total_amount = invoice_data_dict.get('total_amount', '')

    return f"Vendor: {vendor} Invoice Number: {invoice_no} Date: {date} Amount: {total_amount}"

def get_embedding(text_string: str) -> List[float]:
    """
    Generates a dense vector embedding for a given text string.

    Args:
        text_string: The input text string (e.g., from invoice_to_text_representation).

    Returns:
        A list of floats representing the dense vector embedding.
    """
    model = _get_embedding_model() # Get model via the new function
    # Use the initialized SentenceTransformer model to encode the text
    embedding = model.encode(text_string)
    # Convert the resulting NumPy array to a Python list
    return embedding.tolist()

def build_faiss_index(embeddings_list: List[List[float]]) -> Union[faiss.Index, None]:
    """
    Builds a FAISS index from a list of invoice embeddings.

    Args:
        embeddings_list: A list of invoice embeddings (each embedding is a list of floats).

    Returns:
        The populated FAISS index object, or None if the embeddings_list is empty.
    """
    if not embeddings_list:
        print("Warning: embeddings_list is empty. Returning None for FAISS index.")
        return None

    embeddings_np = np.array(embeddings_list).astype('float32')
    
    if embeddings_np.ndim == 1: # Handle case of single embedding gracefully for dimension check
        if embeddings_np.shape[0] == 0:
            print("Warning: embeddings_list resulted in an empty numpy array. Returning None.")
            return None
        # If it's a single embedding, reshape it to be 2D for consistency
        embeddings_np = embeddings_np.reshape(1, -1)
    elif embeddings_np.ndim == 0: # Should not happen with list[list[float]] but good to check
        print("Warning: embeddings_list resulted in a 0-dimensional numpy array. Returning None.")
        return None

    dimension = embeddings_np.shape[1]
    if dimension == 0:
        print("Warning: Embeddings have 0 dimension. Returning None.")
        return None

    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity

    # Normalize embeddings for IndexFlatIP if cosine similarity is desired
    faiss.normalize_L2(embeddings_np)
    
    index.add(embeddings_np)
    return index

def search_faiss_index(index: faiss.Index, query_embedding: List[float], top_k: int) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Searches a FAISS index for the top_k nearest neighbors to a query embedding.

    Args:
        index: The FAISS index object.
        query_embedding: A single query embedding (list of floats).
        top_k: The number of nearest neighbors to retrieve.

    Returns:
        A tuple containing two NumPy arrays: distances (similarity scores) and indices.
        Returns (None, None) if the index is not valid or query_embedding is empty.
    """
    if not index or index.ntotal == 0:
        print("Warning: FAISS index is not valid or empty. Cannot search.")
        return None, None
    
    if not query_embedding:
        print("Warning: Query embedding is empty. Cannot search.")
        return None, None
        
    query_np = np.array([query_embedding]).astype('float32')

    # Normalize query embedding (must match normalization at index time)
    faiss.normalize_L2(query_np)
    
    distances, indices = index.search(query_np, top_k)
    return distances, indices

# Default heuristic thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.88 # Lowered from 0.90 to catch near-identical matches with minor OCR diffs (like invoice_no)
DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD = 1.0 # Absolute difference in currency units
DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD = 7  # Absolute difference in days
DEFAULT_TOP_K_FAISS = 20 # Increased from 5 to 20

def find_potential_duplicates(
    target_invoice_id: str,
    all_invoices_data: list[dict], # Each dict should have 'id', 'total_amount', 'date'
    invoice_embeddings: list[list[float]], # Assumed to be parallel to all_invoices_data
    faiss_index: faiss.Index,
    embedding_id_map: dict[int, str], # Maps FAISS index position to original invoice_id/filename
    top_k: int = DEFAULT_TOP_K_FAISS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    amount_threshold: float = DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD,
    date_threshold_days: int = DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD
) -> List[dict]:
    """
    Finds potential duplicate invoices for a target invoice using FAISS and business heuristics.

    Args:
        target_invoice_id: The ID/filename of the invoice to check.
        all_invoices_data: A list of all invoice dictionaries.
        invoice_embeddings: A list of all invoice embeddings, parallel to all_invoices_data.
        faiss_index: The pre-built FAISS index of all invoice_embeddings.
        embedding_id_map: Dictionary mapping FAISS index position to original invoice_id.
        top_k: Number of nearest neighbors to retrieve from FAISS.
        similarity_threshold: Minimum cosine similarity to consider a match.
        amount_threshold: Maximum absolute difference in total_amount.
        date_threshold_days: Maximum absolute difference in invoice dates (in days).

    Returns:
        A list of potential duplicate invoices.
    """
    potential_duplicates = []

    target_invoice_data = None
    target_invoice_list_idx = -1
    for i, inv_data in enumerate(all_invoices_data):
        if inv_data.get('id') == target_invoice_id or inv_data.get('filename') == target_invoice_id:
            target_invoice_data = inv_data
            target_invoice_list_idx = i
            break

    if target_invoice_data is None:
        print(f"Warning: Target invoice ID '{target_invoice_id}' not found in all_invoices_data.")
        return potential_duplicates

    if not (0 <= target_invoice_list_idx < len(invoice_embeddings)):
        print(f"Warning: Target invoice index {target_invoice_list_idx} for ID '{target_invoice_id}' is out of bounds for invoice_embeddings (len: {len(invoice_embeddings)}). Ensure all_invoices_data and invoice_embeddings are parallel and complete.")
        return potential_duplicates
        
    query_embedding = invoice_embeddings[target_invoice_list_idx]
    if not query_embedding:
        print(f"Warning: Query embedding for '{target_invoice_id}' is empty.")
        return potential_duplicates

    distances, indices = search_faiss_index(faiss_index, query_embedding, top_k)

    if distances is None or indices is None or len(indices[0]) == 0:
        # print(f"Info: FAISS search yielded no results or failed for '{target_invoice_id}'.")
        return potential_duplicates

    for i in range(len(indices[0])):
        faiss_match_idx = indices[0][i]
        # FAISS can return -1 if k is larger than index.ntotal and not enough neighbors are found
        # or for some specific index types/configurations, though IndexFlatIP usually doesn't.
        if faiss_match_idx == -1:
            continue
            
        similarity_score = float(distances[0][i])

        if faiss_match_idx not in embedding_id_map:
            print(f"Warning: FAISS match index {faiss_match_idx} not found in embedding_id_map. Skipping.")
            continue
        matched_invoice_original_id = embedding_id_map[faiss_match_idx]

        if matched_invoice_original_id == target_invoice_id:
            continue

        current_matched_invoice_data = None
        for inv_data in all_invoices_data:
            if inv_data.get('id') == matched_invoice_original_id or inv_data.get('filename') == matched_invoice_original_id:
                current_matched_invoice_data = inv_data
                break
        
        if current_matched_invoice_data is None:
            print(f"Warning: Matched invoice ID '{matched_invoice_original_id}' (from FAISS index {faiss_match_idx}) not found in all_invoices_data. Skipping.")
            continue

        # Apply Heuristics
        if not (similarity_score >= similarity_threshold):
            print(f"DEBUG: Pair ({target_invoice_id}, {matched_invoice_original_id}) failed SIMILARITY check. Score: {similarity_score:.4f} < Threshold: {similarity_threshold}")
            continue

        try:
            target_amount_str = str(target_invoice_data.get('total_amount', '0.0')).replace(',', '')
            matched_amount_str = str(current_matched_invoice_data.get('total_amount', '0.0')).replace(',', '')

            is_target_amount_nan = (target_amount_str.lower() == 'nan')
            is_matched_amount_nan = (matched_amount_str.lower() == 'nan')

            if is_target_amount_nan and is_matched_amount_nan:
                # Both are NaN, consider it a pass for amount heuristic
                pass 
            elif is_target_amount_nan or is_matched_amount_nan:
                # One is NaN, the other is not. Consider it a fail.
                print(f"DEBUG: Pair ({target_invoice_id}, {matched_invoice_original_id}) failed AMOUNT check because one amount is NaN. Target: '{target_amount_str}', Matched: '{matched_amount_str}'")
                continue
            else:
                # Neither is NaN, proceed with float conversion and comparison
                target_amount = float(target_amount_str)
                matched_amount = float(matched_amount_str)
                if not (abs(target_amount - matched_amount) <= amount_threshold):
                    print(f"DEBUG: Pair ({target_invoice_id}, {matched_invoice_original_id}) failed AMOUNT check. Target: {target_amount}, Matched: {matched_amount}, Diff: {abs(target_amount - matched_amount):.2f} > Threshold: {amount_threshold}")
                    continue
        except ValueError: # Catches float conversion errors if non-NaN, non-numeric strings slip through
            print(f"Warning: Could not convert amounts for '{target_invoice_id}' ('{target_amount_str}') or '{matched_invoice_original_id}' ('{matched_amount_str}'). Skipping pair.")
            continue

        try:
            target_date_str = target_invoice_data.get('date')
            matched_date_str = current_matched_invoice_data.get('date')

            # Handle cases where date might be a float 'nan'
            if isinstance(target_date_str, float) and np.isnan(target_date_str):
                target_date_str = None
            if isinstance(matched_date_str, float) and np.isnan(matched_date_str):
                matched_date_str = None

            if not target_date_str or not matched_date_str:
                print(f"DEBUG: Pair ({target_invoice_id}, {matched_invoice_original_id}) skipped DATE check because one or both dates are missing/None. TargetDate: '{target_date_str}', MatchedDate: '{matched_date_str}'")
                # If we skip the date check, the pair might still be a duplicate based on other heuristics.
                # Decide if 'continue' here is the right logic or if it should proceed to be added to potential_duplicates
                # For now, assuming if dates are critical and one is missing, it's not a confident match for this heuristic.
                # However, the original logic was to 'continue' if dates were missing.
                # Let's refine this: if dates are missing, we can't apply the date *difference* heuristic.
                # The pair should still be considered if amount and similarity are met.
                # So, if we are here, it means amount and similarity *were* met.
                # The 'continue' for missing dates was to skip date *parsing* and *comparison*.
                # If we reach here after this block, it implies date check isn't strictly applied if dates are missing.
                # The original logic was:
                # if not target_date_str or not matched_date_str:
                #    continue << THIS IS THE KEY. It would skip the pair if dates were missing.
                # Let's re-evaluate. If the goal is to pass if dates are missing, remove the continue here.
                # If dates *must* be present to be a valid duplicate according to this path, keep continue.
                # Given the previous "Skipping pair on date heuristic." for unparseable dates,
                # it implies the date check is somewhat strict.
                # For now, let's assume if dates are missing, we cannot confirm via date, so we pass it (no 'continue')
                # and let it be added if similarity and amount matched.
                # The date *difference* check later will naturally not run if parsed_target_date or parsed_matched_date is None.

                # Re-evaluating: the original code had:
                # if not target_date_str or not matched_date_str:
                #    continue
                # This means if dates were missing, the pair was SKIPPED.
                # We should keep this behavior for now unless we decide to change the core logic.
                # The DEBUG print above is fine.
                pass # No specific action, fall through to parsing attempt (which will handle Nones)

            parsed_target_date, parsed_matched_date = None, None
            try:
                if target_date_str: # Only parse if not None
                    if isinstance(target_date_str, (str, bytes)):
                        parsed_target_date = parser.parse(str(target_date_str), fuzzy=False)
                    else:
                        print(f"Warning: Target date for '{target_invoice_id}' is not a string ('{target_date_str}', type: {type(target_date_str)}). Skipping date parse.")
                
                if matched_date_str: # Only parse if not None
                    if isinstance(matched_date_str, (str, bytes)):
                        parsed_matched_date = parser.parse(str(matched_date_str), fuzzy=False)
                    else:
                        print(f"Warning: Matched date for '{matched_invoice_original_id}' is not a string ('{matched_date_str}', type: {type(matched_date_str)}). Skipping date parse.")

            except (parser.ParserError, ValueError, TypeError) as e:
                print(f"DEBUG: Pair ({target_invoice_id}, {matched_invoice_original_id}) had PARSING error for dates. Target: '{target_date_str}', Matched: '{matched_date_str}'. Error: {e}. Skipping pair on date heuristic.")
                continue

            if parsed_target_date and parsed_matched_date: # Both dates must be successfully parsed to compare
                day_difference = abs((parsed_target_date - parsed_matched_date).days)
                if not (day_difference <= date_threshold_days):
                    print(f"DEBUG: Pair ({target_invoice_id}, {matched_invoice_original_id}) failed DATE DIFFERENCE check. TargetDate: {parsed_target_date.strftime('%Y-%m-%d') if parsed_target_date else 'N/A'}, MatchedDate: {parsed_matched_date.strftime('%Y-%m-%d') if parsed_matched_date else 'N/A'}, Diff_Days: {day_difference} > Threshold_Days: {date_threshold_days}")
                    continue
            elif target_date_str or matched_date_str: # If at least one date was originally present but couldn't be fully processed (e.g., one parsed, other didn't)
                # This case implies one date was present and parseable, the other was missing or unparseable.
                # Or both present but one failed parsing.
                # This is a stricter check: if we intended to compare dates, but couldn't get both, we fail.
                print(f"DEBUG: Pair ({target_invoice_id}, {matched_invoice_original_id}) skipped DATE COMPARISON because not both dates were successfully parsed. ParsedTarget: {parsed_target_date is not None}, ParsedMatched: {parsed_matched_date is not None}. Original Target: '{target_date_str}', Original Matched: '{matched_date_str}'")
                continue # Strict: if date data is present but not usable for comparison, skip.

        except Exception as e:
            print(f"Warning: Error processing dates/amounts for '{target_invoice_id}', '{matched_invoice_original_id}': {e}. Skipping pair.")
            continue

        potential_duplicates.append({
            'matched_invoice_id': matched_invoice_original_id,
            'similarity_score': similarity_score,
            'details': current_matched_invoice_data
        })

    return potential_duplicates

def get_embeddings(text_list: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> List[List[float]]:
    """
    Encodes a list of strings in a single batch call to the underlying SentenceTransformer
    model. This is much more efficient and avoids repeatedly constructing DataLoaders,
    which on macOS can leak multiprocessing semaphores and eventually crash the
    interpreter.

    Args:
        text_list: List of strings to embed.
        batch_size: Batch size for SentenceTransformer.encode.
        show_progress_bar: Whether to display the tqdm progress bar.

    Returns:
        List of embeddings as lists of floats, in the same order as the input.
    """
    if not text_list:
        return []
    model = _get_embedding_model()
    embeddings_np = model.encode(text_list, batch_size=batch_size, show_progress_bar=show_progress_bar)
    return [vec.tolist() for vec in embeddings_np]

# Example Usage (can be removed or commented out)
if __name__ == '__main__':
    sample_invoice_data = {
        'vendor': 'Tech Solutions Inc.',
        'invoice_no': 'INV-2023-001',
        'date': '2023-10-26',
        'total_amount': '1500.00'
    }
    
    text_rep = invoice_to_text_representation(sample_invoice_data)
    print(f"Text Representation: {text_rep}")
    
    embedding_vector = get_embedding(text_rep)
    print(f"Embedding Vector (first 5 dimensions): {embedding_vector[:5]}")
    print(f"Embedding Dimension: {len(embedding_vector)}")

    sample_invoice_data_missing_fields = {
        'vendor': 'Office Supplies Co.',
        'total_amount': '75.50'
    }
    text_rep_missing = invoice_to_text_representation(sample_invoice_data_missing_fields)
    print(f"Text Representation (missing fields): {text_rep_missing}")
    embedding_vector_missing = get_embedding(text_rep_missing)
    print(f"Embedding Vector (missing fields, first 5 dimensions): {embedding_vector_missing[:5]}") 