import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import streamlit as st

# Call set_page_config as the very first Streamlit command
st.set_page_config(page_title="MVP Invoice Auditor", layout="wide")

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
from typing import Union, Tuple, List, Dict # Import Union, Tuple, List, Dict
from src.graph_manager import connect_to_neo4j, check_invoice_exceeds_contract, fetch_graph_data, load_invoices_to_graph, load_contracts_to_graph, clear_graph # Added graph_manager imports
from src.gnn_compliance_model import InvoiceGNN, export_data_for_gnn, predict_compliance_with_gnn # Added GNN imports
import json
import streamlit.components.v1 as components

# Attempt to import the evaluation script function
# This needs scripts directory to be in PYTHONPATH or use relative import if structure allows
# For simplicity, assuming it can be imported. If not, PYTHONPATH adjustment or path manipulation needed.
try:
    from scripts.evaluate_duplicates import get_duplicate_detection_metrics
except ImportError as e:
    st.sidebar.warning(f"Could not import evaluation script: {e}. Performance dashboard will be unavailable.")
    get_duplicate_detection_metrics = None # Placeholder if import fails

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

def display_3d_graph(data: dict):
    json_data = json.dumps(data)
    html_string = """
    <div id="3d-graph" style="width: 100%; height: 600px;"></div>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <script>
      const Graph = ForceGraph3D()(document.getElementById('3d-graph'))
        .graphData({json_data})
        // Color nodes by risk: red=high, green=low
        .nodeColor(node => {{
          const r = node.risk || 0;
          const red = Math.floor(255 * r);
          const green = Math.floor(255 * (1 - r));
          return `rgb(${{red}},${{green}},0)`;
        }})
        // Show label with risk on hover
        .nodeLabel(node => `${{node.label}} (Risk: ${{(node.risk||0).toFixed(2)}})`)
        // Label links by relationship type
        .linkDirectionalArrowLength(4)
        .linkDirectionalArrowRelPos(1)
        .linkLabel(link => link.type)
        .onNodeClick(node => {{
          alert(`Clicked node: ${{node.label || node.id}}\nRisk: ${{(node.risk||0).toFixed(2)}}`);
        }});
    </script>
    """.format(json_data=json_data)
    components.html(html_string, height=650)

def display_performance_dashboard():
    st.subheader("Duplicate Detection Model Performance")

    if get_duplicate_detection_metrics is None:
        st.error("Performance metric calculation is unavailable due to an import error. Please check the logs.")
        return

    # Use session state to store metrics and avoid recalculating on every rerun
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = None
    if 'calculating_metrics' not in st.session_state:
        st.session_state.calculating_metrics = False

    if st.button("Calculate/Refresh Duplicate Detection Metrics", key="refresh_metrics_button"):
        st.session_state.calculating_metrics = True
        st.session_state.performance_metrics = None # Clear previous metrics

    if st.session_state.calculating_metrics:
        with st.spinner("Calculating performance metrics... This may take a moment."):
            try:
                precision, recall, f1, cm_df = get_duplicate_detection_metrics()
                st.session_state.performance_metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion_matrix": cm_df
                }
                st.success("Performance metrics calculated!")
            except FileNotFoundError as e:
                st.error(f"Error calculating metrics: Required data file not found. {e}")
                st.session_state.performance_metrics = None # Clear on error
            except (ValueError, RuntimeError) as e:
                st.error(f"Error calculating metrics: {e}")
                st.session_state.performance_metrics = None # Clear on error
            except Exception as e:
                st.error(f"An unexpected error occurred while calculating metrics: {e}")
                st.session_state.performance_metrics = None # Clear on error
            finally:
                st.session_state.calculating_metrics = False # Reset flag
    
    if st.session_state.performance_metrics:
        metrics = st.session_state.performance_metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{metrics['precision']:.4f}")
        col2.metric("Recall", f"{metrics['recall']:.4f}")
        col3.metric("F1-score", f"{metrics['f1']:.4f}")

        st.subheader("Confusion Matrix")
        if not metrics['confusion_matrix'].empty:
            st.table(metrics['confusion_matrix'])
        else:
            st.info("Confusion matrix is empty or could not be calculated.")
    elif not st.session_state.calculating_metrics: # Only show if not currently calculating
        st.info("Click the button above to calculate and display performance metrics for the duplicate detection model.")

def main():
    st.title("Invoice Auditing MVP")
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Extract", "Invoice & Compliance Dashboard", "Performance Metrics"] )

    # Page-specific content
    if page == "Upload & Extract":
        st.write("Welcome to the AI-assisted invoice auditing system.")
        # --- Upload & Extraction Logic ---
        api_key_available = bool(OPENAI_API_KEY or GEMINI_API_KEY)
        if not api_key_available:
            st.error("An API key (OpenAI or Gemini) is not configured. Extraction features will be disabled.")
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
                    invoice_id_for_compliance = newly_extracted_data.get("invoice_no")
                    if not invoice_id_for_compliance:
                        st.warning("`invoice_no` not found in extracted data. Compliance check might be inaccurate or fail if it's the required ID in Neo4j.")
                        invoice_id_for_compliance = newly_extracted_data.get('id')

                    if invoice_id_for_compliance:
                        try:
                            with st.spinner("Connecting to Neo4j and checking compliance..."):
                                graph = connect_to_neo4j()
                                if graph:
                                    results = check_invoice_exceeds_contract(graph, invoice_id_for_compliance)
                                else:
                                    results = None
                                    st.error("Failed to connect to Neo4j. Compliance check skipped.")

                            if results is None:
                                pass
                            elif results:
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
                            else:
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

                else: # newly_extracted_data is None or empty
                    if uploaded_file is not None: # only show error if a file was indeed uploaded but extraction failed
                        st.error("Extraction failed. No data to process.")

            except Exception as e:
                st.error(f"An error occurred during extraction or duplicate check: {str(e)}")
                # Consider logging the full exception traceback to the console or a log file for debugging
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}") 
                # For security, don't show full tracebacks in the UI in production.
    elif page == "Invoice & Compliance Dashboard":
        st.header("Invoice & Compliance Dashboard")
        st.write("Explore your invoice, vendor, and contract relationships in 3D.")
        try:
            graph_db = connect_to_neo4j()
            
            # --- Clear existing graph data before loading new data ---
            with st.spinner("Clearing existing graph data..."):
                 clear_graph(graph_db)
                 st.info("Cleared previous graph data from Neo4j.")
            # --- End Clear Graph ---

            # Load actual invoice and contract data into Neo4j
            import pandas as _pd
            invoices_df = _pd.read_csv("data/extracted_invoices.csv")
            # Ensure column 'invoice_id' exists for graph loading
            if 'invoice_no' in invoices_df.columns:
                invoices_df.rename(columns={'invoice_no': 'invoice_id'}, inplace=True)
            elif 'filename' in invoices_df.columns:
                invoices_df.rename(columns={'filename': 'invoice_id'}, inplace=True)
            
            # --- FIX for NaN invoice_id --- 
            if 'invoice_id' in invoices_df.columns:
                original_row_count = len(invoices_df)
                invoices_df.dropna(subset=['invoice_id'], inplace=True)
                invoices_df = invoices_df[invoices_df['invoice_id'].astype(str).str.strip() != '']
                if len(invoices_df) < original_row_count:
                    st.warning(f"Dropped {original_row_count - len(invoices_df)} rows from invoices due to missing/NaN invoice_id.")
            else:
                st.error("Critical error: 'invoice_id' column not found after renaming attempts. Cannot load invoices to graph.")
                return
            # --- END FIX --- 
            
            # --- FIX for NaN vendor in invoices_df ---
            if 'vendor' in invoices_df.columns:
                original_row_count_vendor_inv = len(invoices_df)
                invoices_df.dropna(subset=['vendor'], inplace=True)
                invoices_df = invoices_df[invoices_df['vendor'].astype(str).str.strip() != '']
                if len(invoices_df) < original_row_count_vendor_inv:
                    st.warning(f"Dropped {original_row_count_vendor_inv - len(invoices_df)} rows from invoices due to missing/NaN vendor name.")
            else:
                st.warning("'vendor' column not found in invoices_df. Vendor nodes might not be created correctly for all invoices.")
            # --- END FIX ---

            contracts_df = _pd.read_csv("data/contracts.csv")
            # Also good to check contracts_df for NaN in key fields like 'contract_id' and 'vendor'
            if 'contract_id' in contracts_df.columns:
                contracts_df.dropna(subset=['contract_id'], inplace=True)
            if 'vendor' in contracts_df.columns: # Vendor is used for linking
                original_row_count_vendor_con = len(contracts_df)
                contracts_df.dropna(subset=['vendor'], inplace=True)
                contracts_df = contracts_df[contracts_df['vendor'].astype(str).str.strip() != '']
                if len(contracts_df) < original_row_count_vendor_con:
                    st.warning(f"Dropped {original_row_count_vendor_con - len(contracts_df)} rows from contracts due to missing/NaN vendor name.")
            # We already check for vendor in contracts_df before this, but if it wasn't there initially:
            # elif 'vendor' not in contracts_df.columns: 
            #    st.warning("'vendor' column not found in contracts_df. Vendor nodes might not be created correctly for all contracts.")


            load_invoices_to_graph(graph_db, invoices_df)
            load_contracts_to_graph(graph_db, contracts_df)
            # Fetch the populated graph
            nodes, links = fetch_graph_data(graph_db)
            # augment with risk ratings if GNN available
            gnn_model = load_gnn_model()
            gnn_data = get_gnn_data(graph_db) if gnn_model else None
            
            if gnn_model and gnn_data:
                st.sidebar.info("GNN model and data loaded for risk prediction.")
                # For debugging: Print some info about gnn_data if needed
                # print(f"DEBUG: GNN data node_id_to_idx keys sample: {list(gnn_data.node_id_to_idx.keys())[:5]}")
            elif gnn_model:
                st.sidebar.warning("GNN model loaded, but GNN data export failed. Risk scores will be 0.")
            else:
                st.sidebar.warning("GNN model not loaded. Risk scores will be 0.")

            for node in nodes:
                if node.get('label') == 'Invoice':
                    # Try to get the most reliable invoice ID property first
                    invoice_id_for_gnn = node.get('invoice_id') # This should be the one from the CSV
                    if not invoice_id_for_gnn: # Fallback if 'invoice_id' property is missing on the node
                        invoice_id_for_gnn = node.get('id') # This would be Neo4j's internal ID if 'invoice_id' isn't a property
                        print(f"DEBUG: Invoice node (Neo4j ID: {node.get('id')}) missing 'invoice_id' property, attempting fallback to Neo4j ID for GNN: {invoice_id_for_gnn}")
                    
                    risk = 0.0 # Default to float
                    if gnn_model and gnn_data and invoice_id_for_gnn:
                        # print(f"DEBUG: Predicting risk for invoice ID: '{invoice_id_for_gnn}'") # Console print
                        try:
                            prob = predict_compliance_with_gnn(gnn_model, gnn_data, str(invoice_id_for_gnn)) # Ensure it's a string
                            if prob is not None:
                                risk = float(prob)
                                # print(f"DEBUG: Predicted probability for '{invoice_id_for_gnn}': {risk}") # Console print
                            else:
                                print(f"DEBUG: GNN predict returned None for invoice '{invoice_id_for_gnn}'. Risk set to 0.")
                        except Exception as e_gnn:
                            print(f"ERROR: Exception during GNN prediction for '{invoice_id_for_gnn}': {e_gnn}")
                            # risk remains 0.0
                    elif not invoice_id_for_gnn:
                        print(f"DEBUG: Could not determine a valid invoice ID for GNN prediction for node: {node}")

                    node['risk'] = risk
            display_3d_graph({"nodes": nodes, "links": links})
        except Exception as e:
            st.error(f"Error loading 3D graph data: {e}")
    else:  # Performance Metrics
        st.header("Model Performance Dashboard")
        display_performance_dashboard()

    # Check if Neo4j credentials are set, if not, inform the user.
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        st.sidebar.warning("Neo4j credentials (URI, USER, PASSWORD) are not fully configured in .env. Compliance check features may fail or be disabled if it relies on them at startup (though connect_to_neo4j handles it at runtime).")

    # Add a check for GNN model file existence
    if not os.path.exists(GNN_MODEL_PATH):
        st.sidebar.warning(f"GNN model file not found at {GNN_MODEL_PATH}. GNN prediction feature will be disabled.")

if __name__ == "__main__":
    main() 