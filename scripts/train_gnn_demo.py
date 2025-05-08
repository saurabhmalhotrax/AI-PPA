import sys
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Adjust path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_manager import connect_to_neo4j
from src.gnn_compliance_model import InvoiceGNN, export_data_for_gnn, train_gnn, predict_compliance_with_gnn, NODE_TYPE_INVOICE, NODE_TYPE_CONTRACT, NODE_TYPE_VENDOR
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD # For connection check

# ---- TEMPORARY DEBUG ----
# print(f"DEBUG: NEO4J_URI='{NEO4J_URI}'")
# print(f"DEBUG: NEO4J_USER='{NEO4J_USER}'")
# print(f"DEBUG: NEO4J_PASSWORD='{NEO4J_PASSWORD}'")
# ---- END TEMPORARY DEBUG ----

def visualize_embeddings(model, data, filename="gnn_embeddings.png"):
    """
    Extracts node embeddings from the trained model, runs t-SNE,
    and saves the visualization.
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings. For SAGEConv, we pass x and edge_index to the layers.
        # It's often insightful to get embeddings from an intermediate layer or before the final linear layer.
        # Assuming model.conv2 is the second SAGEConv layer.
        # If your model structure is different (e.g. embeddings are stored differently), adjust this.
        # We need to ensure all nodes get an embedding.
        # The forward pass of the model up to the point before the final classifier can give us this.
        
        # A common way to get embeddings is from the output of the last GNN layer
        # In InvoiceGNN, this would be after self.conv2 and before self.out
        x = model.conv1(data.x, data.edge_index).relu()
        # x = F.dropout(x, p=0.5, training=model.training) # Dropout is usually off during eval
        embeddings = model.conv2(x, data.edge_index).relu()

    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
    embeddings_np = embeddings.numpy()

    if embeddings_np.shape[0] < 2 :
        print("Not enough nodes to run t-SNE (less than 2). Skipping visualization.")
        return

    print(f"Running t-SNE on {embeddings_np.shape[0]} node embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30.0, max(5.0, embeddings_np.shape[0] - 1.0))) # Adjust perplexity
    embeddings_2d = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(12, 10))
    
    # Colors for different node types
    colors = {
        NODE_TYPE_INVOICE: 'blue',
        NODE_TYPE_CONTRACT: 'green',
        NODE_TYPE_VENDOR: 'red'
    }
    labels = {
        NODE_TYPE_INVOICE: 'Invoice',
        NODE_TYPE_CONTRACT: 'Contract',
        NODE_TYPE_VENDOR: 'Vendor'
    }

    # Determine node types for coloring
    # The 'all_nodes_info' should be part of the 'data' object from export_data_for_gnn
    if hasattr(data, 'all_nodes_info') and data.all_nodes_info is not None:
        node_types_np = np.array([node_info['type'] for node_info in data.all_nodes_info])
        
        unique_types = np.unique(node_types_np)

        for node_type_val in unique_types:
            mask = (node_types_np == node_type_val)
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1], 
                c=colors.get(node_type_val, 'gray'), # Default to gray if type not in map
                label=labels.get(node_type_val, f'Type {node_type_val}'), 
                alpha=0.7,
                s=20
            )
    else:
        # Fallback if node type info is not available, plot all as one color
        print("Warning: Node type information for coloring not found in data object. Plotting all nodes in one color.")
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=20)

    plt.title("t-SNE Visualization of GNN Node Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig(filename)
    print(f"t-SNE visualization saved to {filename}")
    plt.close()

def main():
    print("Starting GNN training and visualization script...")

    # 1. Check Neo4j connection details
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        print("Error: Neo4j connection details (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) not found.")
        print("Please set them in your .env file and ensure src/config.py loads them.")
        return

    # 2. Connect to Neo4j
    try:
        print(f"Connecting to Neo4j ({NEO4J_URI})...")
        graph = connect_to_neo4j()
        # Test connection with a simple query
        graph.run("MATCH (n) RETURN count(n) LIMIT 1").data()
        print("Successfully connected to Neo4j.")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    # 3. Export data for GNN
    try:
        print("Exporting data from Neo4j for GNN model...")
        # Assuming export_data_for_gnn returns a PyTorch Geometric Data object
        gnn_data = export_data_for_gnn(graph)
        if gnn_data.num_nodes == 0:
            print("No data exported from Neo4j. Cannot train GNN. Check graph population.")
            return
        if gnn_data.train_mask.sum().item() == 0:
            print("Warning: No nodes in the training set based on train_mask. Training might not be effective.")
            # We can still proceed to see if the model runs, but it won't learn.
        print(f"Data exported: {gnn_data.num_nodes} nodes, {gnn_data.num_edges} edges.")
        print(f"Number of node features: {gnn_data.num_node_features}")
        print(f"Training nodes: {gnn_data.train_mask.sum().item()}, Validation nodes: {gnn_data.val_mask.sum().item()}, Test nodes: {gnn_data.test_mask.sum().item()}")

    except Exception as e:
        print(f"Error exporting data for GNN: {e}")
        return

    # 4. Instantiate InvoiceGNN model
    # The number of features comes from the exported data object
    # hidden_channels can be tuned, e.g., 16 or 32. num_classes is 1 for binary compliance.
    hidden_channels = 32 # Example, can be tuned
    model = InvoiceGNN(num_node_features=gnn_data.num_node_features, hidden_channels=hidden_channels, num_classes=1)
    print(f"Initialized InvoiceGNN model: {model}")

    # 5. Train the GNN model
    epochs = 100 # As per requirement
    print(f"Starting GNN training for {epochs} epochs...")
    try:
        trained_model = train_gnn(model, gnn_data, epochs=epochs) # train_gnn should print metrics
        print("GNN training completed.")
    except Exception as e:
        print(f"Error during GNN training: {e}")
        return

    # 6. Print training and validation accuracy/loss (train_gnn should handle this)
    # If train_gnn doesn't print final metrics, we might need to add a call here
    # For now, assuming train_gnn logs these details during training.
    # We can add a final evaluation step on test data if available and meaningful
    if gnn_data.test_mask.sum().item() > 0:
        trained_model.eval()
        with torch.no_grad():
            out = trained_model(gnn_data.x, gnn_data.edge_index)
            criterion = torch.nn.BCEWithLogitsLoss()
            test_loss = criterion(out[gnn_data.test_mask], gnn_data.y[gnn_data.test_mask])
            preds = (torch.sigmoid(out[gnn_data.test_mask]) > 0.5).float()
            correct = (preds == gnn_data.y[gnn_data.test_mask]).sum().item()
            total = gnn_data.test_mask.sum().item()
            test_acc = correct / total if total > 0 else 0
            print(f"Final Test Set Evaluation: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f} ({correct}/{total})")
    else:
        print("No test set available for final evaluation.")

    # 7. (Optional) Extract node embeddings and visualize with t-SNE
    # Let's make this conditional, e.g., by a command line argument or config in a real scenario.
    # For this task, we'll try to run it if matplotlib and sklearn are available.
    try:
        print("Attempting to generate t-SNE visualization of node embeddings...")
        visualize_embeddings(trained_model, gnn_data, filename="gnn_embeddings.png")
    except ImportError:
        print("Skipping t-SNE visualization: matplotlib or sklearn not installed.")
    except Exception as e:
        print(f"Error during t-SNE visualization: {e}")

    # 8. Save model state dict
    model_save_path = "trained_gnn_model.pt"
    try:
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Trained GNN model state_dict saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # 9. Test prediction for a sample invoice
    print("\n--- Testing GNN Prediction on a Sample Invoice ---")
    if hasattr(gnn_data, 'all_nodes_info') and hasattr(gnn_data, 'node_id_to_idx') and hasattr(gnn_data, 'test_mask') and hasattr(gnn_data, 'y'):
        example_invoice_original_id = None
        test_invoice_node_indices = gnn_data.test_mask.nonzero(as_tuple=True)[0]
        
        if len(test_invoice_node_indices) > 0:
            # Prefer an invoice from the test set
            test_node_idx = test_invoice_node_indices[0].item()
            # Find the original ID corresponding to this test_node_idx
            for node_info_item in gnn_data.all_nodes_info:
                if gnn_data.node_id_to_idx.get(node_info_item['id']) == test_node_idx and node_info_item['type'] == NODE_TYPE_INVOICE:
                    example_invoice_original_id = node_info_item['original_id']
                    break
            if example_invoice_original_id:
                print(f"Selected test invoice ID for prediction: {example_invoice_original_id}")
        
        if not example_invoice_original_id:
            # Fallback: try to find any invoice if no suitable test invoice was found
            print("No test invoice found or issue mapping. Falling back to first available invoice.")
            for node_info_item in gnn_data.all_nodes_info:
                if node_info_item['type'] == NODE_TYPE_INVOICE:
                    example_invoice_original_id = node_info_item['original_id']
                    print(f"Selected fallback invoice ID for prediction: {example_invoice_original_id}")
                    break

        if example_invoice_original_id:
            prediction_prob = predict_compliance_with_gnn(trained_model, gnn_data, example_invoice_original_id)
            if prediction_prob is not None:
                # Find the actual label for this invoice
                node_idx_for_label = gnn_data.node_id_to_idx.get(f"inv_{example_invoice_original_id}")
                if node_idx_for_label is not None:
                    actual_label_val = gnn_data.y[node_idx_for_label].item()
                    print(f"  Actual Label: {'Non-compliant' if actual_label_val == 1.0 else 'Compliant'} (raw: {actual_label_val})")
                else:
                    print(f"  Could not find node index for invoice {example_invoice_original_id} to show actual label.")
        else:
            print("Could not find any example invoice ID to make a prediction.")
    else:
        print("Skipping sample prediction test: gnn_data object is missing required attributes.")

    # 10. Simulate a "separate" prediction run using the SAVED and LOADED model
    print("\n--- Simulating Prediction with Independently Loaded Model ---")
    model_load_path = "trained_gnn_model.pt" # Same as model_save_path
    # Check if essential variables from training are available.
    # gnn_data contains num_node_features and is needed for predict_compliance_with_gnn
    # hidden_channels was used to define the model architecture
    if os.path.exists(model_load_path) and 'gnn_data' in locals() and hasattr(gnn_data, 'num_node_features') and 'hidden_channels' in locals():
        try:
            print(f"Loading model from {model_load_path}...")
            # Re-initialize model structure - ensure these params match the saved model
            loaded_model_instance = InvoiceGNN(num_node_features=gnn_data.num_node_features,
                                               hidden_channels=hidden_channels, # This variable should be in scope from model init
                                               num_classes=1)
            loaded_model_instance.load_state_dict(torch.load(model_load_path))
            loaded_model_instance.eval() # Set to evaluation mode
            print("Model loaded successfully.")

            invoice_id_for_loaded_test = None
            # Try to use the same example_invoice_original_id from the previous test section if it's available
            if 'example_invoice_original_id' in locals() and example_invoice_original_id is not None:
                invoice_id_for_loaded_test = example_invoice_original_id
                print(f"Using previously selected invoice ID for loaded model test: {invoice_id_for_loaded_test}")
            elif hasattr(gnn_data, 'all_nodes_info') and gnn_data.all_nodes_info: # Fallback if not
                for node_info_item in gnn_data.all_nodes_info:
                    if node_info_item['type'] == NODE_TYPE_INVOICE:
                        invoice_id_for_loaded_test = node_info_item['original_id']
                        print(f"Selected fallback invoice ID for loaded model test: {invoice_id_for_loaded_test}")
                        break
            
            if invoice_id_for_loaded_test:
                print(f"Making prediction for invoice ID: {invoice_id_for_loaded_test} using the loaded model.")
                # The predict_compliance_with_gnn function expects the full gnn_data object
                # as it uses it to find the node by original_id and get its features and context.
                prediction_prob_loaded = predict_compliance_with_gnn(loaded_model_instance,
                                                                     gnn_data, # Pass the full gnn_data
                                                                     invoice_id_for_loaded_test)
                if prediction_prob_loaded is not None:
                    # For comparison, show actual label again
                    node_idx_for_label = gnn_data.node_id_to_idx.get(f"inv_{invoice_id_for_loaded_test}")
                    if node_idx_for_label is not None and hasattr(gnn_data, 'y'):
                        actual_label_val = gnn_data.y[node_idx_for_label].item()
                        # predict_compliance_with_gnn already prints the probability
                        # print(f"  Prediction (Prob Non-Compliant): {prediction_prob_loaded:.4f}")
                        print(f"  Actual Label: {'Non-compliant' if actual_label_val == 1.0 else 'Compliant'} (raw: {actual_label_val})")
                    else:
                        print(f"  Could not find node index or labels for invoice {invoice_id_for_loaded_test} to show actual label.")
            else:
                print("Could not find an example invoice ID for the loaded model test.")

        except Exception as e:
            print(f"Error during loaded model test: {e}")
    else:
        if not os.path.exists(model_load_path):
            print(f"Model file {model_load_path} not found. Skipping loaded model test.")
        elif not ('gnn_data' in locals() and hasattr(gnn_data, 'num_node_features')):
            print("Skipping loaded model test: gnn_data or its num_node_features not available.")
        elif not 'hidden_channels' in locals():
            print("Skipping loaded model test: hidden_channels variable not available for model initialization.")
        else:
            print("Skipping loaded model test due to missing prerequisites.")

    print("\nGNN training and visualization script finished.")

if __name__ == "__main__":
    main() 