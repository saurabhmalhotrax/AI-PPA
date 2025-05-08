import pandas as pd
import sys
import os

# Adjust the Python path to include the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from graph_manager import (
    connect_to_neo4j,
    clear_graph,
    load_invoices_to_graph,
    load_contracts_to_graph
)
from config import NEO4J_URI, NEO4J_PASSWORD, NEO4J_USER # To ensure config is loaded

def main():
    """
    Main function to connect to Neo4j, clear existing data,
    and load invoices and contracts from CSV files.
    """
    print("Starting Neo4j data population script...")

    # Define paths to data files
    # Assuming the script is run from the root of the project or `scripts` directory
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    invoices_csv_path = os.path.join(base_path, 'data/extracted_invoices.csv')
    contracts_csv_path = os.path.join(base_path, 'data/contracts.csv')

    # Check if data files exist
    if not os.path.exists(invoices_csv_path):
        print(f"Error: Invoices file not found at {invoices_csv_path}")
        return
    if not os.path.exists(contracts_csv_path):
        print(f"Error: Contracts file not found at {contracts_csv_path}")
        return

    print(f"Loading invoices from: {invoices_csv_path}")
    invoices_df = pd.read_csv(invoices_csv_path)
    print(f"Loaded {len(invoices_df)} invoices.")
    print(f"Debug: Initial invoice_df columns: {list(invoices_df.columns)}")

    # Standardize column names (e.g., strip whitespace, convert to lowercase)
    invoices_df.columns = invoices_df.columns.str.strip().str.lower()
    print(f"Debug: Standardized invoice_df columns: {list(invoices_df.columns)}")

    # Rename 'invoice_no' to 'invoice_id' if it exists
    if 'invoice_no' in invoices_df.columns:
        invoices_df.rename(columns={'invoice_no': 'invoice_id'}, inplace=True)
    elif 'invoice_id' not in invoices_df.columns:
        print("Error: Neither 'invoice_no' nor 'invoice_id' found in invoices CSV.")
        return

    print(f"Loading contracts from: {contracts_csv_path}")
    contracts_df = pd.read_csv(contracts_csv_path)
    print(f"Loaded {len(contracts_df)} contracts.")

    # Standardize contract column names
    print(f"Debug: Initial contract_df columns: {list(contracts_df.columns)}")
    contracts_df.columns = contracts_df.columns.str.strip().str.lower()
    print(f"Debug: Standardized contract_df columns: {list(contracts_df.columns)}")

    # Rename 'contract_no' to 'contract_id' if it exists
    if 'contract_no' in contracts_df.columns:
        print("Debug: Renaming 'contract_no' to 'contract_id' in contracts_df")
        contracts_df.rename(columns={'contract_no': 'contract_id'}, inplace=True)
    elif 'contract_id' not in contracts_df.columns:
        print("Warning: Neither 'contract_no' nor 'contract_id' found in contracts CSV. This may cause issues if 'contract_id' is essential.")

    # Rename 'vendor_name' to 'vendor' if 'vendor_name' exists (based on provided schema)
    if 'vendor_name' in contracts_df.columns:
        print("Debug: Renaming 'vendor_name' to 'vendor' in contracts_df")
        contracts_df.rename(columns={'vendor_name': 'vendor'}, inplace=True)
    elif 'vendor' not in contracts_df.columns:
        print("Warning: Neither 'vendor_name' nor 'vendor' found in contracts CSV after standardization. This is crucial for relationships.")

    try:
        print(f"Connecting to Neo4j at {NEO4J_URI}...")
        graph = connect_to_neo4j()
        print("Successfully connected to Neo4j.")

        print("Clearing existing data from the graph...")
        clear_graph(graph)
        print("Graph cleared.")

        print("Loading invoices to graph...")
        # Ensure required columns exist, otherwise graph_manager will raise ValueError
        # For invoices: invoice_id, vendor, date, total_amount
        # For contracts: contract_id, vendor, max_value, start_date, end_date
        # Example: Convert date columns if they are not string, as py2neo might expect specific types or string representations
        if 'date' in invoices_df.columns:
            invoices_df['date'] = pd.to_datetime(invoices_df['date'], errors='coerce').astype(str)
        if 'start_date' in contracts_df.columns:
            contracts_df['start_date'] = pd.to_datetime(contracts_df['start_date'], errors='coerce').astype(str)
        if 'end_date' in contracts_df.columns:
            contracts_df['end_date'] = pd.to_datetime(contracts_df['end_date'], errors='coerce').astype(str)

        # Enhanced debugging for invoices_df before dropna
        print("--- Pre-dropna debug for invoices_df ---")
        cols_to_check_invoice = ['invoice_id', 'vendor', 'date']
        all_invoice_cols_accessible = True
        for col_name in cols_to_check_invoice:
            if col_name in invoices_df.columns:
                try:
                    print(f"Debug: Accessing invoices_df['{col_name}']. Head:\\n{invoices_df[col_name].head(2)}")
                except Exception as access_e:
                    print(f"Debug: ERROR accessing invoices_df['{col_name}']: {type(access_e).__name__} - {access_e}")
                    all_invoice_cols_accessible = False
            else:
                print(f"Debug: Column '{col_name}' NOT FOUND in invoices_df.columns: {list(invoices_df.columns)}")
                all_invoice_cols_accessible = False
        
        if not all_invoice_cols_accessible:
            print("Error: Not all required columns for invoices_df.dropna were accessible. Aborting.")
            return
        
        print(f"Debug: Columns before invoices_df.dropna: {list(invoices_df.columns)}")
        invoices_df = invoices_df.dropna(subset=cols_to_check_invoice)
        
        print(f"Debug: Columns before contracts_df.dropna: {list(contracts_df.columns)}")
        # Ensure 'vendor' is part of the subset if it's expected after renaming
        contracts_dropna_subset = ['contract_id', 'vendor', 'start_date', 'end_date']
        # Verify all columns in contracts_dropna_subset actually exist in contracts_df before dropna
        missing_contract_cols = [col for col in contracts_dropna_subset if col not in contracts_df.columns]
        if missing_contract_cols:
            print(f"Warning: Columns {missing_contract_cols} not found in contracts_df for dropna. Current columns: {list(contracts_df.columns)}")
            # Adjust subset to only include existing columns to avoid error, or handle as needed
            contracts_dropna_subset = [col for col in contracts_dropna_subset if col in contracts_df.columns]

        if contracts_dropna_subset: # only drop if there are columns to check
             contracts_df = contracts_df.dropna(subset=contracts_dropna_subset)
        else:
            print("Warning: No columns left in subset for contracts_df.dropna. Skipping dropna for contracts.")

        # Filter out invoices/contracts if vendor name is missing, as it's crucial for relationship
        invoices_df = invoices_df[invoices_df['vendor'].notna() & (invoices_df['vendor'] != '')]
        contracts_df = contracts_df[contracts_df['vendor'].notna() & (contracts_df['vendor'] != '')]

        print("Debug: invoices_df columns before loading:", invoices_df.columns)
        print("Debug: invoices_df head before loading:\n", invoices_df.head())
        load_invoices_to_graph(graph, invoices_df)
        print(f"{len(invoices_df)} invoices processed for loading.")

        print("Loading contracts to graph...")
        load_contracts_to_graph(graph, contracts_df)
        print(f"{len(contracts_df)} contracts processed for loading.")

        # Query Neo4j for actual counts post-loading
        invoice_count_query = "MATCH (i:Invoice) RETURN count(i) AS count"
        contract_count_query = "MATCH (c:Contract) RETURN count(c) AS count"
        vendor_count_query = "MATCH (v:Vendor) RETURN count(v) AS count"

        num_invoices_in_db = graph.evaluate(invoice_count_query)
        num_contracts_in_db = graph.evaluate(contract_count_query)
        num_vendors_in_db = graph.evaluate(vendor_count_query)

        print("\n----- Population Summary -----")
        print(f"Invoices loaded into Neo4j: {num_invoices_in_db}")
        print(f"Contracts loaded into Neo4j: {num_contracts_in_db}")
        print(f"Unique vendors created/merged in Neo4j: {num_vendors_in_db}")
        print("-----------------------------")
        print("Data population script finished successfully.")

    except Exception as e:
        print(f"An error occurred (type: {type(e).__name__}): {e}")
        print("Data population script failed.")

if __name__ == "__main__":
    main() 