import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import streamlit as st
import torch # Added torch import for GNN model loading
from src.vision_extractor import extract_invoice_data
# VISION_API_KEY is imported as requested, but OPENAI_API_KEY/GEMINI_API_KEY are used by extract_invoice_data
from src.config import VISION_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD # Added Neo4j config
from src.duplicate_detector import (
    invoice_to_text_representation,
    get_embedding,
    build_faiss_index,
    search_faiss_index, # Changed from find_potential_duplicates
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD,
    DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD,
    DEFAULT_TOP_K_FAISS # Added for search_faiss_index
)
from dateutil import parser as date_parser # For parsing dates
from datetime import datetime, timedelta # For date comparisons
import numpy as np # For isnan
import pandas as pd # Added for st.table if data is a list of dicts or dict of dicts
from typing import Union # Import Union
from src.graph_manager import connect_to_neo4j, check_invoice_exceeds_contract # Added graph_manager imports
from src.gnn_compliance_model import InvoiceGNN, export_data_for_gnn, predict_compliance_with_gnn # Added GNN imports

# Path to the CSV file
EXTRACTED_INVOICES_CSV = "data/extracted_invoices.csv"

# Path to the trained GNN model
GNN_MODEL_PATH = "trained_gnn_model.pt"
HIDDEN_CHANNELS = 32 # Updated to match hidden_channels used when the model was trained
NUM_NODE_FEATURES = 4 # Based on export_data_for_gnn implementation

@st.cache_resource # Cache the GNN model loading
def load_gnn_model(model_path=GNN_MODEL_PATH, num_node_features=NUM_NODE_FEATURES, hidden_channels=HIDDEN_CHANNELS):
    """
    Loads the pre-trained GNN model.
    """
    if not os.path.exists(model_path):
        st.warning(f"GNN model file not found at {model_path}. GNN prediction will be skipped.")
        return None
    try:
        model = InvoiceGNN(num_node_features=num_node_features, hidden_channels=hidden_channels)
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set model to evaluation mode
        st.sidebar.success("GNN model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading GNN model from {model_path}: {e}")
        return None

# Streamlit can't hash py2neo Graph; prefix arg with _ to skip hashing
@st.cache_data
def get_gnn_data(_graph):
    """
    Exports and caches graph data for the GNN.
    """
    try:
        return export_data_for_gnn(_graph)
    except Exception as e:
        st.error(f"Error exporting graph data for GNN: {e}")
        return None

@st.cache_data # Cache the data loading and preparation
def load_and_prepare_existing_invoices():
    """
    Loads existing invoices from CSV, prepares text representations, embeddings,
    and builds a FAISS index.
    Returns:
        Tuple: (all_invoices_data_list, faiss_index, embedding_id_map)
               Returns (None, None, None) if CSV not found or processing fails.
    """
    if not os.path.exists(EXTRACTED_INVOICES_CSV):
        st.error(f"Historical data file not found: {EXTRACTED_INVOICES_CSV}. Duplicate detection will be skipped.")
        return None, None, None

    try:
        df_existing = pd.read_csv(EXTRACTED_INVOICES_CSV)
        if df_existing.empty:
            st.info("No existing invoices found in CSV for duplicate checking.")
            return [], None, {} # Return empty list and None index if df is empty
    except Exception as e:
        st.error(f"Error reading {EXTRACTED_INVOICES_CSV}: {e}")
        return None, None, None

    # Prepare data structures
    all_invoices_data_list: list[dict] = []
    text_reprs: list[str] = []
    embedding_id_map: dict[int, str] = {}

    # Ensure required columns exist (warn but proceed if missing)
    required_cols = ['filename', 'vendor', 'invoice_no', 'date', 'total_amount']
    if not all(col in df_existing.columns for col in required_cols):
        st.error(f"CSV must contain columns: {', '.join(required_cols)}. Please check {EXTRACTED_INVOICES_CSV}.")

    # Build invoice dicts & text representations
    for index, row in df_existing.iterrows():
        invoice_dict = row.to_dict()
        invoice_id = invoice_dict.get('id', invoice_dict.get('filename', f"csv_row_{index}"))
        invoice_dict['id'] = invoice_id
        all_invoices_data_list.append(invoice_dict)
        text_reprs.append(invoice_to_text_representation(invoice_dict))

    # Batch encode all text representations at once
    from src.duplicate_detector import get_embeddings as _batch_embed
    all_embeddings_list = _batch_embed(text_reprs, batch_size=32, show_progress_bar=False)

    # Build embedding_id_map sequentially (FAISS will use the same order)
    for idx, inv in enumerate(all_invoices_data_list):
        embedding_id_map[idx] = inv['id']

    if not all_embeddings_list:
        return all_invoices_data_list, None, embedding_id_map

    faiss_index_cache = build_faiss_index(all_embeddings_list)
    if faiss_index_cache is None:
        st.warning("Failed to build FAISS index for existing invoices.")

    return all_invoices_data_list, faiss_index_cache, embedding_id_map

def parse_date_flexible(date_str: str) -> Union[datetime, None]:
    if pd.isna(date_str) or not date_str:
        return None
    try:
        return date_parser.parse(str(date_str))
    except (ValueError, TypeError):
        return None

def check_invoice_heuristics(
    new_invoice_data: dict,
    existing_invoice_data: dict,
    amount_threshold: float = DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD,
    date_threshold_days: int = DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD
) -> bool:
    """
    Checks amount and date heuristics between two invoices.
    """
    try:
        new_amount_str = str(new_invoice_data.get('total_amount', '0.0')).replace(',', '')
        existing_amount_str = str(existing_invoice_data.get('total_amount', '0.0')).replace(',', '')

        is_new_amount_nan = (new_amount_str.lower() == 'nan')
        is_existing_amount_nan = (existing_amount_str.lower() == 'nan')

        if is_new_amount_nan and is_existing_amount_nan:
            pass # Both NaN, amount check passes
        elif is_new_amount_nan or is_existing_amount_nan:
            return False # One is NaN, fails
        else:
            new_amount = float(new_amount_str)
            existing_amount = float(existing_amount_str)
            if abs(new_amount - existing_amount) > amount_threshold:
                return False
    except ValueError:
        return False # Amount conversion failed

    new_date = parse_date_flexible(new_invoice_data.get('date'))
    existing_date = parse_date_flexible(existing_invoice_data.get('date'))

    if new_date and existing_date:
        if abs((new_date - existing_date).days) > date_threshold_days:
            return False
    elif new_date or existing_date: # One date is missing, stricter check might fail here
        return False # If one date is missing, consider it a fail for now for stricter duplication.

    return True

def main():
    st.set_page_config(page_title="MVP Invoice Auditor", layout="wide")
    st.title("Invoice Auditing MVP") # Simplified title slightly
    st.write("Welcome to the AI-assisted invoice auditing system.")

    # Add a check for API key availability
    # Defaulting to check OPENAI_API_KEY as extract_invoice_data defaults to use_openai=True
    # You might want to refine this logic if you allow selecting between OpenAI and Gemini
    api_key_available = bool(OPENAI_API_KEY or GEMINI_API_KEY) # Check if at least one key is available

    if not api_key_available:
        st.error("An API key (OpenAI or Gemini) is not configured. Please set it in your .env file. Extraction features will be disabled.")
        # Optionally, you could prevent the uploader from showing if no key is available
        # return 

    # Load existing data for duplicate detection
    # This will be cached by Streamlit
    existing_invoices_master_list, faiss_index_cache, embedding_id_map_cache = load_and_prepare_existing_invoices()

    st.header("Upload and Extract Invoice Data")
    uploaded_file = st.file_uploader("Upload Invoice (PDF/Image)", type=["pdf","png","jpg","jpeg"])

    if uploaded_file is not None:
        if not api_key_available:
            st.error("Cannot process file: API key is missing.")
            return

        image_bytes = uploaded_file.read()
        
        # Determine which API to use. Defaulting to OpenAI if its key is available.
        # This can be made more sophisticated with a user toggle.
        use_openai_service = True # Default
        if OPENAI_API_KEY:
            st.info("Using OpenAI for extraction.")
            use_openai_service = True
        elif GEMINI_API_KEY:
            st.info("OpenAI API key not found, attempting to use Gemini for extraction.")
            use_openai_service = False
        else:
            # This case should ideally be caught by the top-level api_key_available check,
            # but as a safeguard:
            st.error("No suitable API key found for extraction (OpenAI or Gemini).")
            return

        newly_extracted_data = None
        try:
            with st.spinner("Extracting fields..."):
                # extract_invoice_data uses OPENAI_API_KEY or GEMINI_API_KEY from config internally
                # It does not take an api_key as a parameter in its current version
                # The second parameter is use_openai (bool)
                newly_extracted_data = extract_invoice_data(image_bytes, use_openai=use_openai_service) 
            
            if newly_extracted_data:
                st.success("Extraction successful!")
                # Displaying data:
                # If data is a flat dictionary, st.json is good.
                # If it's a list of dictionaries or a dictionary of dictionaries, st.table might be better.
                # Assuming data is a dictionary as per extract_invoice_data's return type hint.
                st.subheader("Extracted Data for Uploaded Invoice")
                # Ensure newly_extracted_data has an 'id' for consistency, use filename
                newly_extracted_data['id'] = uploaded_file.name 
                
                # Display new data using both JSON and Table
                st.json(newly_extracted_data)
                if isinstance(newly_extracted_data, dict):
                    df_new = pd.DataFrame(list(newly_extracted_data.items()), columns=['Field', 'Value'])
                    st.table(df_new)

                # --- Duplicate Detection Logic ---
                st.subheader("Duplicate Check")
                if faiss_index_cache is not None and existing_invoices_master_list:
                    new_invoice_text_repr = invoice_to_text_representation(newly_extracted_data)
                    new_invoice_embedding = get_embedding(new_invoice_text_repr)

                    if new_invoice_embedding:
                        # Use DEFAULT_TOP_K_FAISS from duplicate_detector
                        distances, indices = search_faiss_index(faiss_index_cache, new_invoice_embedding, top_k=DEFAULT_TOP_K_FAISS)
                        
                        potential_duplicates_details = []
                        if distances is not None and indices is not None and len(indices[0]) > 0:
                            for i in range(len(indices[0])):
                                faiss_match_idx = indices[0][i]
                                similarity_score = float(distances[0][i])

                                # FAISS can return -1 for invalid indices in some cases
                                if faiss_match_idx == -1 or faiss_match_idx not in embedding_id_map_cache:
                                    continue
                                
                                matched_invoice_original_id = embedding_id_map_cache[faiss_match_idx]
                                
                                # Find the full data for the matched existing invoice
                                matched_existing_invoice_data = next((inv for inv in existing_invoices_master_list if inv['id'] == matched_invoice_original_id), None)

                                if matched_existing_invoice_data:
                                    # Check similarity threshold first (from FAISS search, which is cosine similarity)
                                    if similarity_score >= DEFAULT_SIMILARITY_THRESHOLD:
                                        # Then check business rule heuristics (amount, date)
                                        if check_invoice_heuristics(
                                            newly_extracted_data, 
                                            matched_existing_invoice_data,
                                            DEFAULT_AMOUNT_DIFFERENCE_THRESHOLD,
                                            DEFAULT_DATE_DIFFERENCE_DAYS_THRESHOLD
                                        ):
                                            potential_duplicates_details.append({
                                                "id": matched_invoice_original_id,
                                                "similarity": f"{similarity_score:.4f}",
                                                "vendor": matched_existing_invoice_data.get("vendor", "N/A"),
                                                "date": matched_existing_invoice_data.get("date", "N/A"),
                                                "amount": matched_existing_invoice_data.get("total_amount", "N/A")
                                            })
                        
                        if potential_duplicates_details:
                            st.warning(f"Found {len(potential_duplicates_details)} potential duplicate(s):")
                            # Sort by similarity (highest first)
                            sorted_duplicates = sorted(potential_duplicates_details, key=lambda x: float(x['similarity']), reverse=True)
                            df_duplicates = pd.DataFrame(sorted_duplicates)
                            st.table(df_duplicates[['id', 'similarity', 'vendor', 'date', 'amount']])
                        else:
                            st.info("No potential duplicates detected based on similarity and heuristics.")
                    else:
                        st.error("Could not generate embedding for the new invoice. Skipping duplicate check.")
                elif not existing_invoices_master_list:
                    st.info("No existing invoices loaded. Duplicate check skipped.")
                else: # faiss_index_cache is None but existing_invoices_master_list might exist
                    st.warning("FAISS index for existing invoices not available. Duplicate check could not be performed effectively.")
                # --- End of Duplicate Detection Logic ---

                # --- Compliance Check (Neo4j) ---
                st.subheader("Compliance Check")
                
                # Ensure newly_extracted_data contains an invoice_id
                # It's set earlier using uploaded_file.name, but good to double-check or ensure it matches graph expectations
                invoice_id_for_compliance = newly_extracted_data.get("invoice_no") # Prefer 'invoice_no' if available
                if not invoice_id_for_compliance:
                    # Fallback if 'invoice_no' isn't extracted, or use 'id' if that's what's in the graph
                    # For this example, let's assume 'invoice_no' is the key used in the graph for invoices.
                    # If your graph uses 'filename' or another field from newly_extracted_data['id'], adjust here.
                    st.warning("`invoice_no` not found in extracted data. Compliance check might be inaccurate or fail if it's the required ID in Neo4j.")
                    invoice_id_for_compliance = newly_extracted_data.get('id') # Fallback to filename-based id


                if invoice_id_for_compliance:
                    try:
                        with st.spinner("Connecting to Neo4j and checking compliance..."):
                            graph = connect_to_neo4j()
                            if graph:
                                results = check_invoice_exceeds_contract(graph, invoice_id_for_compliance)
                            else:
                                results = None # Should ideally not happen if connect_to_neo4j raises error on failure
                                st.error("Failed to connect to Neo4j. Compliance check skipped.")
                        
                        if results is None: # Handles case where graph connection failed silently or check_invoice_exceeds_contract had an issue
                            pass # Error already shown or will be handled if results is an empty list
                        elif results: # Non-empty list means violations found
                            st.error(f"Compliance Violations Found for Invoice ID: {invoice_id_for_compliance}")
                            violations_data = []
                            for res in results:
                                violations_data.append({
                                    "Invoice ID": res.get('invoice_id', 'N/A'),
                                    "Invoice Amount": res.get('invoice_amount', 'N/A'),
                                    "Contract ID": res.get('contract_id', 'N/A'),
                                    "Contract Max Value": res.get('contract_value', 'N/A')
                                })
                            st.table(pd.DataFrame(violations_data))
                        else: # Empty list means no violations
                            st.success(f"Invoice ID: {invoice_id_for_compliance} is within contract limits.")
                    
                    except Exception as e:
                        st.error(f"Error during compliance check: {e}")
                else:
                    st.warning("No suitable Invoice ID found in extracted data for compliance check.")

                # --- GNN Compliance Prediction ---
                st.subheader("GNN Compliance Prediction")
                
                gnn_model = load_gnn_model()
                
                if gnn_model:
                    try:
                        with st.spinner("Preparing graph data and predicting compliance with GNN..."):
                            # Need Neo4j connection for GNN data export
                            graph_for_gnn = connect_to_neo4j() 
                            if graph_for_gnn:
                                gnn_data = get_gnn_data(graph_for_gnn)
                                if gnn_data:
                                    # predict_compliance_with_gnn expects original invoice ID
                                    prediction_prob = predict_compliance_with_gnn(gnn_model, gnn_data, invoice_id_for_compliance)
                                    if prediction_prob is not None:
                                        # Display probability (assuming predict_compliance_with_gnn returns a float 0-1)
                                        st.metric("GNN Predicted Non-Compliance Probability", f"{prediction_prob:.2f}")
                                        st.progress(float(prediction_prob), text="Probability of Non-Compliance")
                                    else:
                                        st.warning("GNN prediction failed for this invoice.")
                                else:
                                    st.warning("Could not prepare graph data for GNN prediction.")
                            else:
                                st.error("Failed to connect to Neo4j for GNN data export. GNN prediction skipped.")

                    except Exception as e:
                         st.error(f"Error during GNN prediction: {e}")


                else:
                    st.info("GNN model not loaded. Skipping GNN prediction.")

            else: # newly_extracted_data is None
                st.error("Extraction failed. Cannot proceed with duplicate or compliance checks.")
        except Exception as e:
            st.error(f"An error occurred during extraction or duplicate check: {str(e)}")
            # Consider logging the full exception traceback to the console or a log file for debugging
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}") 
            # For security, don't show full tracebacks in the UI in production.

    # Placeholder for future content (from original app.py)
    # st.sidebar.header("Navigation")
    # page = st.sidebar.radio("Go to", ["Upload & Extract", "View Duplicates", "Compliance Check"])

    # if page == "Upload & Extract":
    # (handled above)

    # Check if Neo4j credentials are set, if not, inform the user.
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        st.sidebar.warning("Neo4j credentials (URI, USER, PASSWORD) are not fully configured in .env. Compliance check features may fail or be disabled if it relies on them at startup (though connect_to_neo4j handles it at runtime).")

    # Add a check for GNN model file existence
    if not os.path.exists(GNN_MODEL_PATH):
        st.sidebar.warning(f"GNN model file not found at {GNN_MODEL_PATH}. GNN prediction feature will be disabled.")

if __name__ == "__main__":
    main() 