# Stage 4: Advanced AI and Polish

## Overview
This final stage focuses on implementing Graph Neural Networks (GNNs) for advanced relationship analysis, adding enterprise-grade features, and deploying the system for production use. We'll enhance the system with advanced AI capabilities, improve the user interface, and ensure the system is ready for enterprise deployment.

## Duration & Investment
- **Timeline**: 5-6 weeks
- **Investment**: ~$100-150 (EC2 + AWS Amplify + optional GNN compute)

## Prerequisites
- Successful completion of Stages 1, 2, and 3
- Functioning Neo4j graph database with invoice and contract relationships
- Operational compliance checking system
- Basic understanding of PyTorch and deep learning concepts

## Step-by-Step Implementation Plan

### Step 4.1: Implement Graph Neural Networks (10-12 days)

#### Tech Stack
- Python 3.9
- PyTorch
- PyTorch Geometric
- Neo4j
- pandas
- AWS EC2

#### Activities
1. **Install PyTorch Geometric and Dependencies**:
   - Install PyTorch Geometric:
   ```bash
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
   ```
   - Install visualization tools:
   ```bash
   pip install matplotlib networkx tensorboard
   ```
   - Test imports:
   ```python
   import torch
   import torch_geometric
   from torch_geometric.nn import GCNConv
   print(f"PyTorch: {torch.__version__}")
   print(f"PyTorch Geometric: {torch_geometric.__version__}")
   ```

2. **Export Graph Data from Neo4j to PyTorch**:
   - Create export functionality:
   ```python
   def export_graph_for_gnn(graph):
       """Export Neo4j graph to PyTorch Geometric format"""
       # Export nodes with features
       invoice_nodes = graph.run("""
       MATCH (i:Invoice)
       RETURN i.invoice_id AS id, i.amount AS amount, 
              toFloat(substring(replace(i.date, '-', ''), 0, 8)) AS date_num, 
              0 AS node_type
       """).data()
       
       vendor_nodes = graph.run("""
       MATCH (v:Vendor)
       RETURN v.name AS id, 0 AS amount, 0 AS date_num, 1 AS node_type
       """).data()
       
       contract_nodes = graph.run("""
       MATCH (c:Contract)
       RETURN c.contract_id AS id, c.value AS amount, 
              toFloat(substring(replace(c.start_date, '-', ''), 0, 8)) AS date_num, 
              2 AS node_type
       """).data()
       
       # Combine nodes and create mapping
       all_nodes = invoice_nodes + vendor_nodes + contract_nodes
       node_mapping = {node['id']: idx for idx, node in enumerate(all_nodes)}
       
       # Create node features tensor
       x = torch.tensor([
           [float(node['amount']), float(node['date_num']), float(node['node_type'])]
           for node in all_nodes
       ], dtype=torch.float)
       
       # Export edges
       issued_by_edges = graph.run("""
       MATCH (i:Invoice)-[:ISSUED_BY]->(v:Vendor)
       RETURN i.invoice_id AS source, v.name AS target, 0 AS edge_type
       """).data()
       
       under_contract_edges = graph.run("""
       MATCH (i:Invoice)-[:UNDER_CONTRACT]->(c:Contract)
       RETURN i.invoice_id AS source, c.contract_id AS target, 1 AS edge_type
       """).data()
       
       with_vendor_edges = graph.run("""
       MATCH (c:Contract)-[:WITH_VENDOR]->(v:Vendor)
       RETURN c.contract_id AS source, v.name AS target, 2 AS edge_type
       """).data()
       
       duplicate_of_edges = graph.run("""
       MATCH (i1:Invoice)-[:DUPLICATE_OF]->(i2:Invoice)
       RETURN i1.invoice_id AS source, i2.invoice_id AS target, 3 AS edge_type
       """).data()
       
       # Combine edges
       all_edges = issued_by_edges + under_contract_edges + with_vendor_edges + duplicate_of_edges
       
       # Create edge index tensor
       edge_index = torch.tensor([
           [node_mapping[edge['source']], node_mapping[edge['target']]]
           for edge in all_edges
       ], dtype=torch.long).t().contiguous()
       
       # Create edge type tensor
       edge_type = torch.tensor([
           edge['edge_type'] for edge in all_edges
       ], dtype=torch.long)
       
       # Get labels for a specific task (e.g., has_compliance_issues)
       labels = graph.run("""
       MATCH (i:Invoice)
       OPTIONAL MATCH (i)-[:UNDER_CONTRACT]->(c:Contract)
       WHERE i.amount > c.value OR date(i.date) < date(c.start_date) OR date(i.date) > date(c.end_date)
       WITH i, count(c) > 0 AS has_issues
       RETURN i.invoice_id AS id, has_issues
       """).data()
       
       y = torch.tensor([
           [1.0 if label['has_issues'] else 0.0]
           for label in labels if label['id'] in node_mapping
       ], dtype=torch.float)
       
       # Create masks for training, validation, testing
       num_nodes = len(all_nodes)
       indices = torch.randperm(num_nodes)
       train_mask = torch.zeros(num_nodes, dtype=torch.bool)
       val_mask = torch.zeros(num_nodes, dtype=torch.bool)
       test_mask = torch.zeros(num_nodes, dtype=torch.bool)
       
       train_mask[indices[:int(0.7 * num_nodes)]] = True
       val_mask[indices[int(0.7 * num_nodes):int(0.85 * num_nodes)]] = True
       test_mask[indices[int(0.85 * num_nodes):]] = True
       
       return {
           'x': x,
           'edge_index': edge_index,
           'edge_type': edge_type,
           'y': y,
           'train_mask': train_mask,
           'val_mask': val_mask,
           'test_mask': test_mask,
           'node_mapping': node_mapping
       }
   ```
   - Test export with sample data
   - Save exported data to disk for reuse

3. **Define GNN Architecture**:
   - Implement GraphSAGE-based GNN model:
   ```python
   import torch
   import torch.nn.functional as F
   from torch_geometric.nn import GraphSAGE
   
   class InvoiceGNN(torch.nn.Module):
       def __init__(self, num_node_features, hidden_channels=64, num_classes=1):
           super(InvoiceGNN, self).__init__()
           # GraphSAGE often performs better than GCN for heterogeneous graphs
           self.conv1 = GraphSAGE(num_node_features, hidden_channels, normalize=True)
           self.conv2 = GraphSAGE(hidden_channels, hidden_channels, normalize=True)
           self.classifier = torch.nn.Linear(hidden_channels, num_classes)
           
       def forward(self, x, edge_index):
           # First Message Passing Layer
           x = self.conv1(x, edge_index)
           x = F.relu(x)
           x = F.dropout(x, p=0.2, training=self.training)
           
           # Second Message Passing Layer
           x = self.conv2(x, edge_index)
           x = F.relu(x)
           
           # Classification Layer
           x = self.classifier(x)
           
           return x
   ```
   - Implement training and evaluation functionality:
   ```python
   def train_gnn(data, model, epochs=100, lr=0.01):
       """Train the GNN model"""
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       criterion = torch.nn.BCEWithLogitsLoss()
       
       # Move to GPU if available
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device)
       data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
       
       # Training loop
       for epoch in range(epochs):
           model.train()
           optimizer.zero_grad()
           
           # Forward pass
           out = model(data['x'], data['edge_index'])
           
           # Calculate loss on training nodes
           loss = criterion(
               out[data['train_mask']].squeeze(),
               data['y'][data['train_mask']].squeeze()
           )
           
           # Backward pass
           loss.backward()
           optimizer.step()
           
           # Validation
           if epoch % 10 == 0:
               model.eval()
               with torch.no_grad():
                   out = model(data['x'], data['edge_index'])
                   val_loss = criterion(
                       out[data['val_mask']].squeeze(),
                       data['y'][data['val_mask']].squeeze()
                   )
                   
                   # Calculate validation accuracy
                   pred = (out[data['val_mask']] > 0).float()
                   correct = (pred.squeeze() == data['y'][data['val_mask']].squeeze()).sum()
                   acc = int(correct) / int(data['val_mask'].sum())
                   
                   print(f'Epoch: {epoch}, Train Loss: {loss.item():.4f}, '
                         f'Val Loss: {val_loss.item():.4f}, Val Acc: {acc:.4f}')
       
       return model
   ```
   - Create visualization functions for GNN interpretation

4. **Train on Labeled Data from Neo4j**:
   - Export graph data using the export function
   - Split data into training, validation, and test sets
   - Train GNN model:
   ```python
   # Create and train the model
   data = export_graph_for_gnn(graph)
   model = InvoiceGNN(num_node_features=data['x'].shape[1])
   model = train_gnn(data, model, epochs=200, lr=0.005)
   
   # Save the model
   torch.save(model.state_dict(), 'invoice_gnn_model.pt')
   ```
   - Evaluate model performance on test set
   - Document accuracy and loss metrics

5. **Implement Node Embedding Visualization**:
   - Create t-SNE visualization of node embeddings:
   ```python
   from sklearn.manifold import TSNE
   import matplotlib.pyplot as plt
   import numpy as np
   
   def visualize_embeddings(model, data):
       """Visualize the learned node embeddings"""
       model.eval()
       with torch.no_grad():
           # Get embeddings from second-to-last layer
           embeddings = model.conv2(model.conv1(data['x'], data['edge_index']), data['edge_index'])
           
       # Convert to numpy for t-SNE
       embeddings = embeddings.cpu().numpy()
       
       # Apply t-SNE dimensionality reduction
       tsne = TSNE(n_components=2, random_state=42)
       reduced_embeddings = tsne.fit_transform(embeddings)
       
       # Create node type masks
       node_types = data['x'][:, 2].cpu().numpy()
       invoice_mask = node_types == 0
       vendor_mask = node_types == 1
       contract_mask = node_types == 2
       
       # Plot embeddings colored by node type
       plt.figure(figsize=(12, 10))
       plt.scatter(reduced_embeddings[invoice_mask, 0], reduced_embeddings[invoice_mask, 1], 
                  c='blue', label='Invoice', alpha=0.7, s=10)
       plt.scatter(reduced_embeddings[vendor_mask, 0], reduced_embeddings[vendor_mask, 1], 
                  c='red', label='Vendor', alpha=0.7, s=30)
       plt.scatter(reduced_embeddings[contract_mask, 0], reduced_embeddings[contract_mask, 1], 
                  c='green', label='Contract', alpha=0.7, s=20)
       
       plt.legend()
       plt.title('t-SNE Visualization of Node Embeddings')
       plt.savefig('node_embeddings.png', dpi=300)
       plt.close()
       
       return reduced_embeddings
   ```
   - Generate visualizations to identify clusters and patterns
   - Create heatmap of attention weights

6. **Optimize GNN for Specific Tasks**:
   - Fine-tune for compliance prediction
   - Implement hyperparameter tuning:
   ```python
   def hyperparameter_tuning(data, hidden_dims=[32, 64, 128], learning_rates=[0.001, 0.005, 0.01]):
       """Tune hyperparameters for the GNN model"""
       best_val_acc = 0
       best_params = None
       
       for hidden_dim in hidden_dims:
           for lr in learning_rates:
               print(f"Training with hidden_dim={hidden_dim}, lr={lr}")
               
               # Initialize model
               model = InvoiceGNN(num_node_features=data['x'].shape[1], hidden_channels=hidden_dim)
               
               # Train
               trained_model = train_gnn(data, model, epochs=100, lr=lr)
               
               # Evaluate
               trained_model.eval()
               with torch.no_grad():
                   out = trained_model(data['x'], data['edge_index'])
                   pred = (out[data['val_mask']] > 0).float()
                   correct = (pred.squeeze() == data['y'][data['val_mask']].squeeze()).sum()
                   val_acc = int(correct) / int(data['val_mask'].sum())
               
               print(f"Validation accuracy: {val_acc:.4f}")
               
               if val_acc > best_val_acc:
                   best_val_acc = val_acc
                   best_params = {'hidden_dim': hidden_dim, 'lr': lr}
       
       print(f"Best parameters: {best_params}, Best validation accuracy: {best_val_acc:.4f}")
       return best_params
   ```
   - Evaluate impact of different graph structures

7. **Integrate GNN with Flask API**:
   - Create prediction endpoint:
   ```python
   @app.route('/api/gnn-predict', methods=['POST'])
   def gnn_predict():
       try:
           # Get invoice data from request
           invoice_data = request.json
           
           # Convert to graph representation
           # This would involve creating a subgraph with the new invoice
           # and its connections to vendors, contracts, etc.
           graph_data = convert_invoice_to_graph(invoice_data)
           
           # Run through GNN model
           predictions = run_gnn_inference(graph_data)
           
           return jsonify({
               "invoice_id": invoice_data["invoice_number"],
               "predictions": {
                   "compliance_risk": float(predictions["compliance_risk"]),
                   "duplicate_probability": float(predictions["duplicate_probability"]),
                   "anomaly_score": float(predictions["anomaly_score"])
               }
           })
       except Exception as e:
           error_id = log_error(e, traceback.format_exc())
           return jsonify({"error": "GNN prediction failed", "error_id": error_id}), 500
   ```
   - Implement batch prediction for multiple invoices
   - Test API with sample invoices

8. **Integrate GNN Predictions into UI**:
   - Update React components to display GNN predictions
   - Add risk scoring visualizations
   - Add tooltips explaining GNN predictions

#### Success Criteria
- PyTorch Geometric successfully integrated with Neo4j
- GNN model trained with at least 90% accuracy on test set
- Visualization of node embeddings showing clear clustering
- GNN predictions integrated into API and UI
- Performance benchmarks documented for GNN inference

### Step 4.2: Add Enterprise Features (7-8 days)

#### Tech Stack
- AWS Cognito
- React
- Flask
- S3
- AWS CloudWatch

#### Activities
1. **Set Up User Authentication**:
   - Create AWS Cognito user pool:
   ```bash
   aws cognito-idp create-user-pool \
     --pool-name InvoiceSystemUserPool \
     --auto-verify-attributes email \
     --schema '[{"Name":"email","Required":true},{"Name":"name","Required":true}]' \
     --policies '{"PasswordPolicy":{"MinimumLength":8,"RequireUppercase":true,"RequireLowercase":true,"RequireNumbers":true,"RequireSymbols":false}}' \
     --mfa-configuration OFF
   ```
   - Create app client:
   ```bash
   aws cognito-idp create-user-pool-client \
     --user-pool-id YOUR_USER_POOL_ID \
     --client-name invoice-system-client \
     --no-generate-secret \
     --explicit-auth-flows ALLOW_USER_SRP_AUTH ALLOW_REFRESH_TOKEN_AUTH
   ```
   - Set up identity pool for AWS resource access
   - Configure hosted UI for login (optional)

2. **Implement Authentication in React**:
   - Install AWS Amplify: `npm install aws-amplify`
   - Configure Amplify in app:
   ```javascript
   // src/index.js
   import { Amplify } from 'aws-amplify';
   
   Amplify.configure({
     Auth: {
       region: 'us-west-2',
       userPoolId: 'YOUR_USER_POOL_ID',
       userPoolWebClientId: 'YOUR_CLIENT_ID',
       mandatorySignIn: true
     }
   });
   ```
   - Create authentication context:
   ```javascript
   // src/contexts/AuthContext.js
   import React, { createContext, useState, useEffect, useContext } from 'react';
   import { Auth } from 'aws-amplify';
   
   const AuthContext = createContext();
   
   export function AuthProvider({ children }) {
     const [currentUser, setCurrentUser] = useState(null);
     const [loading, setLoading] = useState(true);
     
     useEffect(() => {
       const checkUser = async () => {
         try {
           const userData = await Auth.currentAuthenticatedUser();
           setCurrentUser(userData);
         } catch (error) {
           setCurrentUser(null);
         } finally {
           setLoading(false);
         }
       };
       
       checkUser();
     }, []);
     
     function signUp(email, password, name) {
       return Auth.signUp({
         username: email,
         password,
         attributes: { email, name }
       });
     }
     
     function signIn(email, password) {
       return Auth.signIn(email, password);
     }
     
     function signOut() {
       return Auth.signOut();
     }
     
     function resetPassword(email) {
       return Auth.forgotPassword(email);
     }
     
     const value = {
       currentUser,
       signUp,
       signIn,
       signOut,
       resetPassword,
       loading
     };
     
     return (
       <AuthContext.Provider value={value}>
         {!loading && children}
       </AuthContext.Provider>
     );
   }
   
   export function useAuth() {
     return useContext(AuthContext);
   }
   ```
   - Create login, signup, and password reset pages
   - Add route protection with authentication HOC

3. **Add Export Functionality**:
   - Implement CSV export:
   ```javascript
   function exportToCsv(invoices) {
     // Define CSV headers
     const headers = ['Invoice Number', 'Date', 'Vendor', 'Amount', 'Status'];
     
     // Convert invoice data to CSV format
     const csvData = invoices.map(invoice => [
       invoice.invoice_number,
       invoice.date,
       invoice.vendor,
       invoice.amount,
       invoice.status
     ]);
     
     // Combine headers and data
     const csvContent = [
       headers.join(','),
       ...csvData.map(row => row.join(','))
     ].join('\n');
     
     // Create blob and download link
     const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
     const url = URL.createObjectURL(blob);
     const link = document.createElement('a');
     link.href = url;
     link.setAttribute('download', `invoices_export_${new Date().toISOString()}.csv`);
     document.body.appendChild(link);
     link.click();
     document.body.removeChild(link);
   }
   ```
   - Implement PDF export using jsPDF:
   ```javascript
   import jsPDF from 'jspdf';
   import 'jspdf-autotable';
   
   function exportToPdf(invoices) {
     const doc = new jsPDF();
     
     // Add title
     doc.setFontSize(18);
     doc.text('Invoice Export Report', 14, 22);
     doc.setFontSize(12);
     doc.text(`Generated on ${new Date().toLocaleDateString()}`, 14, 30);
     
     // Create table
     doc.autoTable({
       startY: 40,
       head: [['Invoice #', 'Date', 'Vendor', 'Amount', 'Status']],
       body: invoices.map(invoice => [
         invoice.invoice_number,
         invoice.date,
         invoice.vendor,
         `$${invoice.amount}`,
         invoice.status
       ])
     });
     
     // Save the PDF
     doc.save(`invoices_export_${new Date().toISOString()}.pdf`);
   }
   ```
   - Add export buttons to the UI

4. **Implement Feedback System**:
   - Create feedback form component:
   ```jsx
   const FeedbackForm = () => {
     const [feedback, setFeedback] = useState('');
     const [category, setCategory] = useState('general');
     const [submitted, setSubmitted] = useState(false);
     const [submitting, setSubmitting] = useState(false);
     
     const handleSubmit = async (e) => {
       e.preventDefault();
       setSubmitting(true);
       
       try {
         const response = await fetch('/api/feedback', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify({ feedback, category })
         });
         
         if (response.ok) {
           setSubmitted(true);
           setFeedback('');
           setCategory('general');
         } else {
           alert('Failed to submit feedback');
         }
       } catch (error) {
         console.error('Error submitting feedback:', error);
         alert('An error occurred while submitting feedback');
       } finally {
         setSubmitting(false);
       }
     };
     
     return (
       <div className="p-4 bg-white rounded shadow">
         <h2 className="text-xl font-semibold mb-4">Your Feedback</h2>
         
         {submitted ? (
           <div className="text-green-600 mb-4">
             Thank you for your feedback!
           </div>
         ) : (
           <form onSubmit={handleSubmit}>
             <div className="mb-4">
               <label className="block mb-2">Category</label>
               <select
                 value={category}
                 onChange={(e) => setCategory(e.target.value)}
                 className="w-full p-2 border rounded"
               >
                 <option value="general">General</option>
                 <option value="extraction">Data Extraction</option>
                 <option value="duplicates">Duplicate Detection</option>
                 <option value="compliance">Compliance Checking</option>
                 <option value="ui">User Interface</option>
               </select>
             </div>
             
             <div className="mb-4">
               <label className="block mb-2">Your Feedback</label>
               <textarea
                 value={feedback}
                 onChange={(e) => setFeedback(e.target.value)}
                 className="w-full p-2 border rounded"
                 rows={5}
                 required
               />
             </div>
             
             <button
               type="submit"
               disabled={submitting}
               className="px-4 py-2 bg-blue-500 text-white rounded"
             >
               {submitting ? 'Submitting...' : 'Submit Feedback'}
             </button>
           </form>
         )}
       </div>
     );
   };
   ```
   - Add feedback API endpoint in Flask
   - Store feedback in database or S3

5. **Add Error Logging and Monitoring**:
   - Set up S3 bucket for error logs
   - Implement error logging service:
   ```python
   import boto3
   import json
   import uuid
   import traceback
   from datetime import datetime
   
   class ErrorLogger:
       def __init__(self, bucket_name, environment='production'):
           self.s3 = boto3.client('s3')
           self.bucket_name = bucket_name
           self.environment = environment
       
       def log_error(self, error, trace=None, context=None):
           """Log error details to S3"""
           # Generate unique error ID
           error_id = str(uuid.uuid4())
           
           # Create error report
           error_report = {
               'error_id': error_id,
               'timestamp': datetime.utcnow().isoformat(),
               'environment': self.environment,
               'error_type': type(error).__name__,
               'error_message': str(error),
               'traceback': trace or traceback.format_exc(),
               'context': context or {}
           }
           
           # Upload to S3
           self.s3.put_object(
               Bucket=self.bucket_name,
               Key=f"errors/{error_id}.json",
               Body=json.dumps(error_report, indent=2),
               ContentType='application/json'
           )
           
           return error_id
   ```
   - Integrate with Flask error handlers
   - Set up AWS CloudWatch alarms

6. **Implement Data Encryption**:
   - Set up S3 server-side encryption
   - Implement transit encryption with HTTPS
   - Add field-level encryption for sensitive data:
   ```python
   from cryptography.fernet import Fernet
   import base64
   import os
   
   class FieldEncryptor:
       def __init__(self, key=None):
           """Initialize with a key or generate a new one"""
           if key:
               self.key = key
           else:
               self.key = Fernet.generate_key()
           self.cipher = Fernet(self.key)
       
       def encrypt(self, text):
           """Encrypt a text string"""
           if not text:
               return None
           return self.cipher.encrypt(text.encode()).decode()
       
       def decrypt(self, encrypted_text):
           """Decrypt an encrypted string"""
           if not encrypted_text:
               return None
           return self.cipher.decrypt(encrypted_text.encode()).decode()
       
       def encrypt_dict(self, data, sensitive_fields):
           """Encrypt sensitive fields in a dictionary"""
           result = data.copy()
           for field in sensitive_fields:
               if field in result and result[field]:
                   result[field] = self.encrypt(str(result[field]))
           return result
       
       def decrypt_dict(self, data, sensitive_fields):
           """Decrypt sensitive fields in a dictionary"""
           result = data.copy()
           for field in sensitive_fields:
               if field in result and result[field]:
                   result[field] = self.decrypt(result[field])
           return result
   ```
   - Document security measures

7. **Add Audit Trail**:
   - Implement activity logging:
   ```python
   class AuditTrail:
       def __init__(self, db_connection):
           self.db = db_connection
       
       def log_activity(self, user_id, action, resource_type, resource_id, details=None):
           """Log user activity in the system"""
           query = """
           INSERT INTO audit_logs (user_id, action, resource_type, resource_id, details, timestamp)
           VALUES (%s, %s, %s, %s, %s, NOW())
           """
           self.db.execute(query, (user_id, action, resource_type, resource_id, json.dumps(details)))
       
       def get_user_activity(self, user_id, limit=100):
           """Get recent activity for a user"""
           query = """
           SELECT * FROM audit_logs
           WHERE user_id = %s
           ORDER BY timestamp DESC
           LIMIT %s
           """
           return self.db.query(query, (user_id, limit))
       
       def get_resource_activity(self, resource_type, resource_id, limit=100):
           """Get activity for a specific resource"""
           query = """
           SELECT * FROM audit_logs
           WHERE resource_type = %s AND resource_id = %s
           ORDER BY timestamp DESC
           LIMIT %s
           """
           return self.db.query(query, (resource_type, resource_id, limit))
   ```
   - Create audit trail UI
   - Implement audit report generation

#### Success Criteria
- AWS Cognito authentication integrated in UI and API
- Export functionality working for CSV and PDF formats
- Feedback system capturing and storing user input
- Error logging system storing detailed error reports
- Data encryption implemented for sensitive information
- Audit trail recording user activities

### Step 4.3: Deploy and Finalize (7-8 days)

#### Tech Stack
- Docker
- AWS ECS
- AWS Amplify
- AWS Route 53 (if needed)
- AWS CloudWatch

#### Activities
1. **Dockerize Flask API**:
   - Create Dockerfile:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       python3-dev \
       poppler-utils \
       libpq-dev \
       && rm -rf /var/lib/apt/lists/*
   
   # Install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Run with gunicorn
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
   ```
   - Create docker-compose.yml for local testing:
   ```yaml
   version: '3'
   
   services:
     api:
       build: .
       ports:
         - "5000:5000"
       environment:
         - FLASK_ENV=production
         - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
         - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
         - AWS_REGION=${AWS_REGION}
         - PINECONE_API_KEY=${PINECONE_API_KEY}
         - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
         - NEO4J_URI=${NEO4J_URI}
         - NEO4J_USER=${NEO4J_USER}
         - NEO4J_PASSWORD=${NEO4J_PASSWORD}
       volumes:
         - ./logs:/app/logs
   ```
   - Test Docker build locally:
   ```bash
   docker-compose build
   docker-compose up
   ```

2. **Deploy to AWS ECS**:
   - Create ECR repository:
   ```bash
   aws ecr create-repository --repository-name invoice-system-api
   ```
   - Push Docker image to ECR:
   ```bash
   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
   docker tag invoice-system-api:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/invoice-system-api:latest
   docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/invoice-system-api:latest
   ```
   - Create ECS task definition
   - Create ECS service
   - Configure auto-scaling based on CPU/memory usage

3. **Deploy React UI to Amplify**:
   - Build production version of React app:
   ```bash
   npm run build
   ```
   - Initialize Amplify hosting:
   ```bash
   amplify init
   amplify add hosting
   ```
   - Deploy to Amplify:
   ```bash
   amplify publish
   ```
   - Configure custom domain (if needed)
   - Set up CI/CD pipeline for automatic deployment

4. **Set Up Monitoring**:
   - Configure CloudWatch dashboards:
   ```bash
   aws cloudwatch put-dashboard --dashboard-name InvoiceSystemDashboard --dashboard-body file://dashboard.json
   ```
   - Set up alarms for critical metrics:
   ```bash
   aws cloudwatch put-metric-alarm \
     --alarm-name api-high-error-rate \
     --alarm-description "API error rate is too high" \
     --metric-name 5XXError \
     --namespace AWS/ApiGateway \
     --statistic Sum \
     --period 300 \
     --threshold 5 \
     --comparison-operator GreaterThanThreshold \
     --evaluation-periods 1 \
     --alarm-actions arn:aws:sns:us-west-2:YOUR_AWS_ACCOUNT_ID:invoice-system-alerts
   ```
   - Configure CloudWatch Logs for API logging
   - Set up email notifications for alarms

5. **Test Deployed System**:
   - Create test plan with key functionality tests
   - Test with 100 new invoices from RVL-CDIP
   - Perform load testing:
   ```bash
   # Using Apache Bench for simple load testing
   ab -n 1000 -c 50 https://api.example.com/api/process-invoice
   ```
   - Address any deployment issues

6. **Create Documentation**:
   - Create system architecture documentation
   - Create user guide with screenshots
   - Document API endpoints:
   ```markdown
   # API Documentation
   
   ## Authentication
   
   All API endpoints require authentication. Include the JWT token in the Authorization header:
   
   ```
   Authorization: Bearer <token>
   ```
   
   ## Endpoints
   
   ### Process Invoice
   
   **URL:** `/api/process-invoice`
   **Method:** POST
   **Content-Type:** multipart/form-data
   
   **Parameters:**
   - `file`: The invoice file (PDF/image)
   
   **Response:**
   ```json
   {
     "invoice_number": "INV-12345",
     "date": "2023-05-15",
     "vendor": "Acme Corp",
     "amount": 1234.56,
     "extracted_fields": {...}
   }
   ```
   
   **Error Response:**
   ```json
   {
     "error": "Processing failed",
     "error_id": "abc123def456"
   }
   ```
   ```
   - Document deployment and operation procedures

7. **Plan Future Enhancements**:
   - Create roadmap for v2 features
   - Document technical debt and improvement areas
   - Schedule regular system reviews and updates

#### Success Criteria
- Flask API successfully dockerized and deployed to ECS
- React UI deployed to Amplify
- System passing all functionality tests
- Monitoring and alerting configured
- Documentation completed
- Future enhancements roadmap created

## Deliverables
1. GNN implementation for relationship analysis
2. Enterprise features (authentication, export, feedback)
3. Error logging and monitoring system
4. Dockerized and deployed API
5. Deployed React UI
6. System documentation and user guides
7. Future enhancements roadmap

## Dependencies and Risks

### Dependencies
- Successful completion of Stages 1-3
- Quality of graph data from Neo4j
- AWS account with appropriate permissions
- Docker and containerization expertise

### Risks
- GNN performance may be lower than expected
- Deployment issues in AWS environment
- Security vulnerabilities in authentication
- Scaling issues with larger datasets

## Mitigation Strategies
- Use simpler GNN architecture if performance is an issue
- Test deployment in staging environment before production
- Follow AWS security best practices and documentation
- Design for scalability with pagination and batch processing
- Include thorough error handling and recovery mechanisms

## Next Steps after Project Completion
- Collect user feedback and prioritize improvements
- Consider ERP integration options
- Explore advanced AI enhancements (e.g., reinforcement learning)
- Plan for SOC 2 compliance if required by enterprise users
- Set up continuous improvement cycle 