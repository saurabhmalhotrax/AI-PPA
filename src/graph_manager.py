from py2neo import Graph, Node, Relationship
import pandas as pd
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def connect_to_neo4j():
    """
    Connect to Neo4j and return the Graph object.
    """
    return Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def clear_graph(graph):
    """
    Remove all nodes and relationships from the graph.
    """
    graph.run("MATCH (n) DETACH DELETE n")

def load_invoices_to_graph(graph, invoices_df: pd.DataFrame):
    """
    Load invoices into Neo4j graph.
    invoices_df must have columns: invoice_id, vendor, date, total_amount
    """
    required_cols = {"invoice_id", "vendor", "date", "total_amount"}
    if not required_cols.issubset(invoices_df.columns):
        raise ValueError(f"Invoices DataFrame must contain columns: {required_cols}")
    for _, row in invoices_df.iterrows():
        invoice_id = row["invoice_id"]
        vendor_name = row["vendor"]
        date = row["date"]
        total_amount = row["total_amount"]

        # Merge Vendor node
        vendor_node = Node("Vendor", name=vendor_name)
        graph.merge(vendor_node, "Vendor", "name")

        # Merge Invoice node
        invoice_node = Node(
            "Invoice",
            invoice_id=invoice_id,
            vendor=vendor_name,
            date=str(date),
            total_amount=total_amount
        )
        graph.merge(invoice_node, "Invoice", "invoice_id")

        # Create relationship
        rel = Relationship(invoice_node, "ISSUED_BY", vendor_node)
        graph.merge(rel)

def load_contracts_to_graph(graph, contracts_df: pd.DataFrame):
    """
    Load contracts into Neo4j graph.
    contracts_df must have columns: contract_id, vendor, max_value, start_date, end_date
    """
    required_cols = {"contract_id", "vendor", "max_value", "start_date", "end_date"}
    if not required_cols.issubset(contracts_df.columns):
        raise ValueError(f"Contracts DataFrame must contain columns: {required_cols}")
    for _, row in contracts_df.iterrows():
        contract_id = row["contract_id"]
        vendor_name = row["vendor"]
        max_value = row["max_value"]
        start_date = row["start_date"]
        end_date = row["end_date"]

        # Merge Vendor node
        vendor_node = Node("Vendor", name=vendor_name)
        graph.merge(vendor_node, "Vendor", "name")

        # Merge Contract node
        contract_node = Node(
            "Contract",
            contract_id=contract_id,
            vendor=vendor_name,
            max_value=max_value,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        graph.merge(contract_node, "Contract", "contract_id")

        # Create relationship
        rel = Relationship(contract_node, "HAS_CONTRACT", vendor_node)
        graph.merge(rel)

def check_invoice_exceeds_contract(graph, invoice_id):
    """
    Checks if a specific invoice's total amount exceeds the max_value of an associated contract.
    Returns a list of dictionaries, each representing a contract that is exceeded.
    """
    query = '''
    MATCH (i:Invoice {invoice_id: $invoice_id})-[:ISSUED_BY]->(:Vendor)<-[:HAS_CONTRACT]-(c:Contract)
    WHERE i.total_amount > c.max_value
    RETURN i.invoice_id AS invoice_id, i.total_amount AS invoice_amount,
           c.contract_id AS contract_id, c.max_value AS contract_value
    '''
    return graph.run(query, invoice_id=invoice_id).data() 