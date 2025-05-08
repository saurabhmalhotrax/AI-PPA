import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv # GraphSAGE layer
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configure multiprocessing for PyTorch on macOS if needed
import multiprocessing as mp
# Attempt to set the start method to 'spawn' if not already set or if on macOS where 'fork' causes issues.
# Guard against NoneType returned by get_start_method to prevent TypeError.
try:
    current_start_method = mp.get_start_method(allow_none=True)
except RuntimeError:
    current_start_method = None  # If the context isn't set yet

# Only attempt to change if we're running as a script OR the current method is missing/"fork".
if __name__ == "__main__" or current_start_method in (None, "fork"):
    try:
        mp.set_start_method("spawn", force=True)
        print("INFO: Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        # This can occur if the start method has already been set elsewhere.
        print(
            f"INFO: Could not set multiprocessing start method to 'spawn': {e}. Current method: {mp.get_start_method(allow_none=True)}"
        )

# Assuming connect_to_neo4j is in graph_manager.py in the same src directory
# If your project structure is different, you might need to adjust the import path
# For scripts, we often add sys.path.append, but for library code, direct relative imports are preferred if structured correctly.
from .graph_manager import connect_to_neo4j, check_invoice_exceeds_contract

# Define Node Types (for feature engineering and masking)
NODE_TYPE_INVOICE = 0
NODE_TYPE_CONTRACT = 1
NODE_TYPE_VENDOR = 2

class InvoiceGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=1):
        super(InvoiceGNN, self).__init__()
        torch.manual_seed(42) # For reproducibility
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2. Apply a final classifier
        x = self.out(x)
        return x

def export_data_for_gnn(graph):
    """
    Exports data from Neo4j and prepares it for the GNN.
    Returns a PyTorch Geometric Data object.
    """
    print("Exporting data from Neo4j for GNN...")

    # 1. Fetch nodes
    invoices_query = "MATCH (n:Invoice) RETURN n.invoice_id AS id, n.total_amount AS amount"
    contracts_query = "MATCH (n:Contract) RETURN n.contract_id AS id, n.max_value AS max_val"
    vendors_query = "MATCH (n:Vendor) RETURN n.name AS id" # Assuming vendor name is unique ID

    invoices_data = graph.run(invoices_query).to_data_frame()
    contracts_data = graph.run(contracts_query).to_data_frame()
    vendors_data = graph.run(vendors_query).to_data_frame()

    print(f"Fetched {len(invoices_data)} invoices, {len(contracts_data)} contracts, {len(vendors_data)} vendors.")

    all_nodes = []
    node_id_to_idx = {}
    current_idx = 0

    # Process invoices
    for _, row in invoices_data.iterrows():
        node_id = f"inv_{row['id']}" # Prefix to ensure uniqueness across types
        if node_id not in node_id_to_idx:
            raw_amount = row.get('amount')
            value = float(0.0 if raw_amount is None or pd.isna(raw_amount) else raw_amount)
            all_nodes.append({'id': node_id, 'original_id': row['id'], 'type': NODE_TYPE_INVOICE, 'value': value})
            node_id_to_idx[node_id] = current_idx
            current_idx += 1
    
    # Process contracts
    for _, row in contracts_data.iterrows():
        node_id = f"con_{row['id']}"
        if node_id not in node_id_to_idx:
            raw_max_val = row.get('max_val')
            value = float(0.0 if raw_max_val is None or pd.isna(raw_max_val) else raw_max_val)
            all_nodes.append({'id': node_id, 'original_id': row['id'], 'type': NODE_TYPE_CONTRACT, 'value': value})
            node_id_to_idx[node_id] = current_idx
            current_idx += 1

    # Process vendors
    for _, row in vendors_data.iterrows():
        node_id = f"ven_{row['id']}"
        if node_id not in node_id_to_idx:
            all_nodes.append({'id': node_id, 'original_id': row['id'], 'type': NODE_TYPE_VENDOR, 'value': 0.0}) # Vendors have no primary numeric value for this feature
            node_id_to_idx[node_id] = current_idx
            current_idx += 1
            
    num_total_nodes = len(all_nodes)
    if num_total_nodes == 0:
        raise ValueError("No nodes found in the graph. Cannot build GNN data.")

    # 2. Create feature matrix (x)
    # Features: [primary_value, is_invoice, is_contract, is_vendor]
    num_node_features = 4 
    x = torch.zeros((num_total_nodes, num_node_features), dtype=torch.float)
    invoice_indices = []

    for idx, node_info in enumerate(all_nodes):
        x[idx, 0] = node_info['value']
        if node_info['type'] == NODE_TYPE_INVOICE:
            x[idx, 1] = 1.0
            invoice_indices.append(idx)
        elif node_info['type'] == NODE_TYPE_CONTRACT:
            x[idx, 2] = 1.0
        elif node_info['type'] == NODE_TYPE_VENDOR:
            x[idx, 3] = 1.0
            
    # 3. Fetch edges and create edge_index
    # Invoice-Vendor relationships
    inv_ven_query = "MATCH (i:Invoice)-[:ISSUED_BY]->(v:Vendor) RETURN i.invoice_id AS inv_id, v.name AS ven_id"
    # Contract-Vendor relationships
    con_ven_query = "MATCH (c:Contract)-[:HAS_CONTRACT]->(v:Vendor) RETURN c.contract_id AS con_id, v.name AS ven_id"
    
    inv_ven_edges = graph.run(inv_ven_query).to_data_frame()
    con_ven_edges = graph.run(con_ven_query).to_data_frame()

    edge_list = []
    for _, row in inv_ven_edges.iterrows():
        source_node_id = f"inv_{row['inv_id']}"
        target_node_id = f"ven_{row['ven_id']}"
        if source_node_id in node_id_to_idx and target_node_id in node_id_to_idx:
            edge_list.append([node_id_to_idx[source_node_id], node_id_to_idx[target_node_id]])
            edge_list.append([node_id_to_idx[target_node_id], node_id_to_idx[source_node_id]]) # Add reverse edges for undirected GNN

    for _, row in con_ven_edges.iterrows():
        source_node_id = f"con_{row['con_id']}"
        target_node_id = f"ven_{row['ven_id']}"
        if source_node_id in node_id_to_idx and target_node_id in node_id_to_idx:
            edge_list.append([node_id_to_idx[source_node_id], node_id_to_idx[target_node_id]])
            edge_list.append([node_id_to_idx[target_node_id], node_id_to_idx[source_node_id]])

    if not edge_list:
        print("Warning: No edges found to build edge_index. GNN might not learn effectively.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 4. Create labels (y) for invoice nodes
    # y = 1 if invoice amount > contract max_value, else 0
    # Only invoice nodes will have meaningful labels for this task.
    y = torch.zeros(num_total_nodes, dtype=torch.float) # BCEWithLogitsLoss expects float
    
    # Use a list to store original invoice IDs for mask creation
    invoice_original_ids_for_mask = []
    
    for node_idx, node_info in enumerate(all_nodes):
        if node_info['type'] == NODE_TYPE_INVOICE:
            invoice_original_ids_for_mask.append(node_info['original_id'])
            # Check compliance using the existing graph_manager function
            # This is simpler than re-querying, but ensure graph_manager.py is accessible
            try:
                exceeding_data = check_invoice_exceeds_contract(graph, node_info['original_id'])
                if exceeding_data: # If the list is not empty, it means it exceeds at least one contract
                    y[node_idx] = 1.0
            except Exception as e:
                print(f"Warning: Could not check compliance for invoice {node_info['original_id']}: {e}")
                # Default to compliant (0) or handle as per requirements if check fails

    # 5. Create train/validation/test masks for INVOICE nodes
    # We only care about predicting compliance for invoices
    num_invoice_nodes = len(invoice_indices)
    train_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_total_nodes, dtype=torch.bool)

    if num_invoice_nodes > 0:
        # Create temporary indices relative to the list of invoice_indices
        temp_indices = np.arange(num_invoice_nodes)
        if num_invoice_nodes < 3: # Not enough data to split
             print("Warning: Very few invoice nodes (<3) to create train/val/test splits. Using all for training.")
             train_indices_rel = temp_indices
             val_indices_rel = []
             test_indices_rel = []
        else:
            train_indices_rel, test_val_indices_rel = train_test_split(temp_indices, test_size=0.4, random_state=42) # 60% train
            if len(test_val_indices_rel) < 2: # Not enough for val and test
                val_indices_rel = test_val_indices_rel
                test_indices_rel = []
            else:
                val_indices_rel, test_indices_rel = train_test_split(test_val_indices_rel, test_size=0.5, random_state=42) # 50% of remaining for val, 50% for test

        # Map relative indices back to global indices
        actual_train_indices = [invoice_indices[i] for i in train_indices_rel]
        actual_val_indices = [invoice_indices[i] for i in val_indices_rel]
        actual_test_indices = [invoice_indices[i] for i in test_indices_rel]

        if actual_train_indices: train_mask[actual_train_indices] = True
        if actual_val_indices: val_mask[actual_val_indices] = True
        if actual_test_indices: test_mask[actual_test_indices] = True
    else:
        print("Warning: No invoice nodes found to create masks.")

    data = Data(x=x, edge_index=edge_index, y=y.unsqueeze(1)) # y needs to be [N, 1] for BCEWithLogitsLoss
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.node_id_to_idx = node_id_to_idx # Store mapping for later predictions
    data.all_nodes_info = all_nodes # Store all node info for lookups

    print("Data export for GNN complete.")
    print(f"  Number of nodes: {data.num_nodes}")
    print(f"  Number of edges: {data.num_edges}")
    print(f"  Number of invoice nodes: {num_invoice_nodes}")
    print(f"  Number of training invoice nodes: {data.train_mask.sum().item()}")
    print(f"  Number of validation invoice nodes: {data.val_mask.sum().item()}")
    print(f"  Number of test invoice nodes: {data.test_mask.sum().item()}")
    
    # Check if any training nodes exist
    if data.train_mask.sum().item() == 0 and num_invoice_nodes > 0:
        print("Warning: No nodes assigned to the training set. Training might not occur.")

    return data


def train_gnn(model, data, epochs=100, lr=0.01):
    """
    Trains the GNN model.
    """
    if data.train_mask.sum().item() == 0:
        print("No training data available (train_mask is all False). Skipping training.")
        return model # Return the un-trained model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss() # Suitable for binary classification (output is raw logits)

    print(f"Starting GNN training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        
        # Only calculate loss on training nodes that are invoices
        # The masks already point to invoice nodes used for training
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Evaluate on validation set (invoice nodes only)
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                # Calculate accuracy for validation invoice nodes
                preds = (torch.sigmoid(val_out[data.val_mask]) > 0.5).float()
                correct = (preds == data.y[data.val_mask]).sum().item()
                total = data.val_mask.sum().item()
                acc = correct / total if total > 0 else 0
                print(f"Epoch {epoch:03d}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}")
    
    print("GNN training finished.")
    return model

def predict_compliance_with_gnn(model, data, invoice_original_id):
    """
    Predicts compliance probability for a specific invoice ID using the trained GNN.
    `data` is the PyG Data object returned by export_data_for_gnn.
    """
    model.eval() # Set model to evaluation mode

    # Find the internal index for the given invoice_original_id
    target_node_idx = None
    # Use the stored mapping and all_nodes_info
    prefixed_invoice_id = f"inv_{invoice_original_id}"
    if prefixed_invoice_id in data.node_id_to_idx:
        target_node_idx = data.node_id_to_idx[prefixed_invoice_id]
        # Verify it's actually an invoice node type from all_nodes_info (optional safety check)
        # node_info = next((n for n in data.all_nodes_info if n['id'] == prefixed_invoice_id), None)
        # if not node_info or node_info['type'] != NODE_TYPE_INVOICE:
        # target_node_idx = None # Reset if type mismatch

    if target_node_idx is None:
        print(f"Error: Invoice ID '{invoice_original_id}' not found in the graph data provided to GNN.")
        return None

    with torch.no_grad():
        all_node_logits = model(data.x, data.edge_index)
        invoice_logit = all_node_logits[target_node_idx]
        probability_non_compliant = torch.sigmoid(invoice_logit).item() # Sigmoid to get probability

    print(f"Predicted probability of invoice '{invoice_original_id}' being non-compliant: {probability_non_compliant:.4f}")
    # Interpretation: Higher probability means more likely to be non-compliant (exceeding contract)
    return probability_non_compliant

# Example Usage (Illustrative - typically run from another script)
if __name__ == '__main__':
    print("Running GNN Compliance Model example...")
    # This example requires a running Neo4j instance with data populated by populate_neo4j.py
    # and NEO4J_PASSWORD, NEO4J_USER, NEO4J_URI set in config or .env
    
    # 1. Connect to Neo4j (ensure config.py can find NEO4J variables)
    try:
        from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
            print("Neo4j connection details not found in src/config.py (likely missing in .env). Exiting.")
            exit()
        graph_db = connect_to_neo4j()
        print("Successfully connected to Neo4j for GNN example.")
    except ImportError:
        print("Could not import Neo4j config. Ensure src/config.py exists and .env is set up.")
        print("Also ensure you are running this from a context where 'src.' imports work (e.g., project root with python -m src.gnn_compliance_model)")
        exit()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        exit()

    # 2. Export data
    try:
        gnn_data = export_data_for_gnn(graph_db)
        
        if gnn_data.num_nodes == 0 or gnn_data.train_mask.sum().item() == 0:
            print("Insufficient data or no training samples after export. GNN cannot be trained. Exiting.")
            exit()

    except ValueError as ve:
        print(f"Error during data export: {ve}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data export: {e}")
        exit()


    # 3. Initialize and train the GNN model
    # num_node_features is derived from how 'x' is constructed in export_data_for_gnn
    # It's [primary_value, is_invoice, is_contract, is_vendor] -> 4 features
    num_features = gnn_data.num_node_features 
    model = InvoiceGNN(num_node_features=num_features, hidden_channels=16)
    
    print(f"Initialized GNN model: {model}")

    trained_model = train_gnn(model, gnn_data, epochs=50, lr=0.01) # Reduced epochs for quick example

    # 4. Make a prediction (example)
    # Find an actual invoice ID from your data to test
    # For this example, let's try to pick one from the test set if available, or any invoice.
    test_invoice_node_indices = gnn_data.test_mask.nonzero(as_tuple=True)[0]
    
    example_invoice_original_id = None
    if len(test_invoice_node_indices) > 0:
        # Get the original ID of the first test invoice
        test_node_idx = test_invoice_node_indices[0].item()
        node_info = next((n for n in gnn_data.all_nodes_info if gnn_data.node_id_to_idx[n['id']] == test_node_idx and n['type'] == NODE_TYPE_INVOICE), None)
        if node_info:
             example_invoice_original_id = node_info['original_id']
    
    if not example_invoice_original_id: # Fallback if no test invoices or issue getting ID
        first_invoice_info = next((n for n in gnn_data.all_nodes_info if n['type'] == NODE_TYPE_INVOICE), None)
        if first_invoice_info:
            example_invoice_original_id = first_invoice_info['original_id']

    if example_invoice_original_id:
        print(f"\nMaking prediction for example invoice ID: {example_invoice_original_id}")
        prediction = predict_compliance_with_gnn(trained_model, gnn_data, example_invoice_original_id)
        if prediction is not None:
            # You can compare this prediction with the actual label in gnn_data.y for that node
            target_node_idx = gnn_data.node_id_to_idx[f"inv_{example_invoice_original_id}"]
            actual_label = gnn_data.y[target_node_idx].item()
            print(f"  Actual label for {example_invoice_original_id}: {'Non-compliant' if actual_label == 1.0 else 'Compliant'}")
            print(f"  Prediction (prob. non-compliant): {prediction:.4f} -> {'Likely Non-compliant' if prediction > 0.5 else 'Likely Compliant'}")
    else:
        print("\nCould not find an example invoice ID to make a prediction.")

    print("\nFinished GNN Compliance Model example.") 