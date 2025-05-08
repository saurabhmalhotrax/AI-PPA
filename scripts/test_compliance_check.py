import sys
import os

# Adjust the Python path to include the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from graph_manager import connect_to_neo4j, check_invoice_exceeds_contract
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD # To ensure config is loaded

def main():
    print("Starting compliance check test script...")

    # Example invoice ID to test.
    # You might want to pick one that you know has a contract and
    # either does or doesn't exceed its max_value for thorough testing.
    test_invoice_id = "1901" 

    print(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
    try:
        graph = connect_to_neo4j()
        print("Successfully connected to Neo4j.")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {type(e).__name__} - {e}")
        print("Please ensure Neo4j is running and .env file has correct credentials (especially NEO4J_PASSWORD).")
        return

    print(f"Checking compliance for invoice_id: {test_invoice_id}")
    try:
        exceeded_contracts = check_invoice_exceeds_contract(graph, test_invoice_id)
        if exceeded_contracts:
            print(f"Invoice '{test_invoice_id}' EXCEEDS contract terms:")
            for contract_info in exceeded_contracts:
                print(f"  - Exceeds Contract ID: {contract_info['contract_id']}, Amount: {contract_info['invoice_amount']} > Max Value: {contract_info['contract_value']}")
        else:
            print(f"Invoice '{test_invoice_id}' is WITHIN contract terms or has no applicable contract for this check.")
    except Exception as e:
        print(f"An error occurred during compliance check: {type(e).__name__} - {e}")

    print("Compliance check test script finished.")

if __name__ == "__main__":
    main() 