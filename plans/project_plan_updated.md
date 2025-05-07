# Invoice Processing System Project Plan

## Project Overview
This project aims to build an enterprise-grade invoice processing system using AI technologies to extract data, detect duplicates, ensure compliance, and analyze invoice relationships. The system leverages DeepSeek-VL for extraction, Pinecone for vector-based deduplication, Neo4j for graph relationships, and Graph Neural Networks (GNNs) for advanced analysis.

## Project Timeline & Investment
- **Duration**: 5-6 months (21-24 weeks)
- **Total Investment**: ~$250-450
  - AWS EC2/S3: ~$200
  - Freelance expertise: ~$50
  - Additional services: ~$50-200

## Key Dataset Integrations

### 1. RVL-CDIP Dataset
- 25,000 invoice images for training and evaluation
- Pre-processed and consistently sized (max dimension 1000px)
- Additional 15 document classes (letters, forms, emails, etc.) enabling document-type filtering
- Accessible via Hugging Face's datasets library
- Research-backed quality from Ryerson Vision Lab

### 2. CUAD (Contract Understanding Atticus Dataset)
- 510 commercial contracts with expert annotations
- 13,000+ labels across 41 categories of important legal clauses
- Designed specifically for contract review in corporate transactions
- Curated by legal experts
- NeurIPS 2021 validated quality
- Licensed under CC BY 4.0

## Staged Implementation Approach

### Stage 1: Core Extraction and Setup
**Duration**: 4-5 weeks  
**Goal**: Extract invoice/contract data with DeepSeek-VL, set up infrastructure, and build a basic UI.  
**Investment**: ~$50-100 (AWS beyond Free Tier if needed)  

#### Step 1.1: Define Scope and Objectives
**Tech**: Google Docs  
**Activities**:
- 1.1.1: Create a Google Doc titled "Accounting System Scope"
- 1.1.2: Define full system capabilities: "System detects duplicates, prevents errors, ensures compliance using DeepSeek-VL, Pinecone, Neo4j, GNNs"
- 1.1.3: List Stage 1 goals: "Process invoices, extract fields, basic UI"
- 1.1.4: Note full vision (vectors, GNNs, enterprise features) for later stages
- 1.1.5: Review and finalize scope document

#### Step 1.2: Set Up Development Environment
**Tech**: VS Code, CursorAI, Python 3.9, Git, AWS (EC2 c6i.large, S3)  
**Activities**:
- 1.2.1: Install VS Code locally
- 1.2.2: Add CursorAI extension
- 1.2.3: Install Python 3.9, Git
- 1.2.4: Sign up for AWS, launch c6i.large (4 vCPUs, 8GB RAM, ~$0.34/hour)
- 1.2.5: Create S3 bucket "invoices-full-system"
- 1.2.6: SSH into EC2, use CursorAI to set up Python project with git
- 1.2.7: Watch "Python Basics" (1 hour), "AWS EC2 Intro" (30 mins)

#### Step 1.3: Gather and Prepare Data
**Tech**: Python 3.9, pandas, Hugging Face datasets  
**Activities**:
- 1.3.1: **[UPDATED]** Load RVL-CDIP dataset's 25,000 invoice images instead of manually sourcing invoices
- 1.3.2: **[UPDATED]** Download and preprocess CUAD's 510 annotated contracts
- 1.3.3: Upload sample data to S3 for testing, save locally too
- 1.3.4: **[NEW]** Create a document classification module to filter non-invoice documents using RVL-CDIP classes

#### Step 1.4: Extract Data with DeepSeek-VL
**Tech**: Python 3.9, PyTorch, bitsandbytes, pdf2image, AWS EC2 c6i.large  
**Activities**:
- 1.4.1: Install packages on EC2: `pip install torch transformers bitsandbytes pdf2image boto3 pandas datasets`
- 1.4.2: Load quantized DeepSeek-VL 7B (from Hugging Face)
- 1.4.3: Use CursorAI to create script: "Convert PDFs to images, run DeepSeek-VL with prompt 'Extract invoice number, amount, supplier, notes'"
- 1.4.4: Process 1,000 RVL-CDIP invoices (~1-2s each on c6i.large), save JSON output
- 1.4.5: Parse JSON to CSV with pandas, upload to S3
- 1.4.6: Test extraction on 50 diverse invoices from RVL-CDIP, evaluate accuracy, adjust prompts as needed

**Sample Code (DeepSeek-VL Optimized Loading):**
```python
# Advanced quantization for better memory efficiency
from transformers import AutoProcessor, AutoModelForVisionLanguageModeling
import torch
from bitsandbytes.nn import Linear4bit

# Load model with 4-bit quantization
processor = AutoProcessor.from_pretrained("deepseek-ai/deepseek-vl-7b-chat")
model = AutoModelForVisionLanguageModeling.from_pretrained(
    "deepseek-ai/deepseek-vl-7b-chat",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    device_map="auto"
)
```

#### Step 1.5: Build Basic UI
**Tech**: React, Node.js, Flask  
**Activities**:
- 1.5.1: Install Node.js locally: `nvm install 18`
- 1.5.2: Use CursorAI to generate React app with upload button and results table
- 1.5.3: On EC2, install Flask: `pip install flask`
- 1.5.4: Use CursorAI to build Flask API to process uploaded invoice with DeepSeek-VL
- 1.5.5: Test locally: upload 1 invoice, view results
- 1.5.6: Add robust error handling to Flask API, test crash recovery scenarios

**Sample Code (Flask API with Error Handling):**
```python
from flask import Flask, request, jsonify
import traceback
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/api/process-invoice', methods=['POST'])
def process_invoice():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        # Process with DeepSeek-VL
        result = extract_from_invoice(file)
        return jsonify(result)
    except Exception as e:
        error_id = log_error(e, traceback.format_exc())
        return jsonify({"error": "Processing failed", "error_id": error_id}), 500

def log_error(exception, traceback_str):
    error_id = generate_unique_id()
    logging.error(f"Error ID: {error_id}")
    logging.error(f"Exception: {str(exception)}")
    logging.error(traceback_str)
    return error_id
```

### Stage 2: Vector-Based Deduplication
**Duration**: 4-5 weeks  
**Goal**: Integrate Pinecone for duplicate detection and refine extraction  
**Investment**: ~$50-100 (EC2 + Pinecone paid tier if Free Tier exceeded)  

#### Step 2.1: Enhance Data Extraction
**Tech**: Python 3.9, PyTorch, spaCy  
**Activities**:
- 2.1.1: Install spaCy: `pip install spacy && python -m spacy download en_core_web_sm`
- 2.1.2: **[UPDATED]** Refine DeepSeek-VL outputs with spaCy, using RVL-CDIP diversity to improve extraction robustness
- 2.1.3: Label 100 invoice pairs (duplicate Y/N) from RVL-CDIP dataset, export CSV
- 2.1.4: Test enhanced extraction on 50 invoices, evaluate improvement

#### Step 2.2: Set Up Vector Database
**Tech**: Pinecone, Python 3.9  
**Activities**:
- 2.2.1: Sign up for Pinecone, create index (768 dims)
- 2.2.2: Install: `pip install pinecone-client sentence-transformers`
- 2.2.3: Use CursorAI to generate script: "Generate embeddings from DeepSeek-VL JSON with SentenceTransformers, upsert to Pinecone"
- 2.2.4: Test upsert with 100 invoices from RVL-CDIP
- 2.2.5: Manually verify 10 vectors, check similarity calculations

**Sample Code (Pinecone Integration):**
```python
from sentence_transformers import SentenceTransformer
import pinecone
import os

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment='us-west1-gcp')
index = pinecone.Index('invoice-index')

def invoice_to_text(invoice_data):
    """Convert invoice data to a single text string for embedding"""
    return f"Invoice {invoice_data['invoice_number']} from {invoice_data['vendor']} for {invoice_data['amount']} dated {invoice_data['date']}"

def embed_and_store(invoice_data):
    """Embed invoice text and store in Pinecone"""
    try:
        invoice_text = invoice_to_text(invoice_data)
        embedding = model.encode(invoice_text).tolist()
        
        # Store in Pinecone with metadata
        index.upsert([(
            invoice_data['invoice_number'], 
            embedding, 
            {
                'vendor': invoice_data['vendor'],
                'amount': invoice_data['amount'],
                'date': invoice_data['date']
            }
        )])
        return True
    except Exception as e:
        print(f"Error embedding invoice {invoice_data['invoice_number']}: {str(e)}")
        return False
```

#### Step 2.3: Implement Duplicate Detection
**Tech**: PyTorch, scikit-learn, Pinecone  
**Activities**:
- 2.3.1: Use CursorAI to generate script: "Compute cosine similarity on Pinecone vectors for labeled pairs"
- 2.3.2: Set threshold (e.g., 0.9), test on 50 pairs, log accuracy (~85% goal)
- 2.3.3: Integrate with Flask API, update React UI to show duplicates
- 2.3.4: If accuracy < 85%, enhance with machine learning model (e.g., Logistic Regression)

### Stage 3: Graph and Compliance
**Duration**: 4-5 weeks  
**Goal**: Build Neo4j graph for compliance and relationships  
**Investment**: ~$50-100 (EC2)  

#### Step 3.1: Set Up Memory Graph
**Tech**: Neo4j Community, Python 3.9, py2neo  
**Activities**:
- 3.1.1: Install Neo4j on EC2: `sudo apt install neo4j`
- 3.1.2: Install py2neo: `pip install py2neo`
- 3.1.3: Define graph schema: nodes (Invoice, Supplier, Contract), edges (duplicate_of, linked_to)
- 3.1.3.1: **[NEW]** Map CUAD's 41 clause categories to Neo4j graph schema for enhanced relationship modeling
- 3.1.4: Use CursorAI to generate script: "Load invoice/contract CSV to Neo4j graph"
- 3.1.5: Run basic queries: `MATCH (i:Invoice)-[:duplicate_of]->(j) RETURN i,j`
- 3.1.6: Identify slow queries, research optimization, implement improvements

**Sample Code (Neo4j Graph Schema with CUAD Integration):**
```python
from py2neo import Graph, Node, Relationship
import pandas as pd

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Clear existing data (for testing)
graph.run("MATCH (n) DETACH DELETE n")

# Create constraints for performance
graph.run("CREATE CONSTRAINT invoice_id IF NOT EXISTS FOR (i:Invoice) REQUIRE i.invoice_id IS UNIQUE")
graph.run("CREATE CONSTRAINT vendor_name IF NOT EXISTS FOR (v:Vendor) REQUIRE v.name IS UNIQUE")
graph.run("CREATE CONSTRAINT clause_type IF NOT EXISTS FOR (c:Clause) REQUIRE c.type IS UNIQUE")

def create_invoice_graph(invoice_df, contracts_df, duplicate_pairs, cuad_clauses):
    """Create graph from invoice data, contracts and known duplicates"""
    # Create invoice nodes
    tx = graph.begin()
    
    vendors = {}
    invoices = {}
    contracts = {}
    clauses = {}
    
    # Create CUAD clause type nodes
    for clause_type in cuad_clauses:
        clause = Node("Clause", type=clause_type)
        tx.create(clause)
        clauses[clause_type] = clause
    
    # Create vendor nodes
    for vendor_name in invoice_df['vendor'].unique():
        vendor = Node("Vendor", name=vendor_name)
        tx.create(vendor)
        vendors[vendor_name] = vendor
    
    # Create invoice nodes and vendor relationships
    for _, row in invoice_df.iterrows():
        invoice = Node(
            "Invoice", 
            invoice_id=row['invoice_number'],
            amount=float(row['amount']),
            date=row['date']
        )
        tx.create(invoice)
        invoices[row['invoice_number']] = invoice
        
        # Connect invoice to vendor
        vendor = vendors[row['vendor']]
        tx.create(Relationship(invoice, "ISSUED_BY", vendor))
    
    # Create contracts and connect to relevant clauses
    for _, row in contracts_df.iterrows():
        contract = Node(
            "Contract",
            contract_id=row['contract_id'],
            vendor=row['vendor'],
            value=float(row['value'])
        )
        tx.create(contract)
        contracts[row['contract_id']] = contract
        
        # Connect contract to vendor
        if row['vendor'] in vendors:
            vendor = vendors[row['vendor']]
            tx.create(Relationship(contract, "WITH_VENDOR", vendor))
        
        # Connect contract to relevant clause types
        for clause_type in row['clauses']:
            if clause_type in clauses:
                tx.create(Relationship(contract, "CONTAINS_CLAUSE", clauses[clause_type]))
    
    # Create duplicate relationships
    for id1, id2 in duplicate_pairs:
        if id1 in invoices and id2 in invoices:
            duplicate_rel = Relationship(invoices[id1], "DUPLICATE_OF", invoices[id2])
            tx.create(duplicate_rel)
    
    tx.commit()
    return len(invoices), len(contracts)
```

#### Step 3.2: Add Compliance Checker
**Tech**: Python 3.9, Neo4j, pandas  
**Activities**:
- 3.2.1: **[UPDATED]** Define compliance rules based on CUAD clause categories
- 3.2.2: Query Neo4j for contract links, test on 50 invoices
- 3.2.3: Update Flask API/UI with compliance alerts
- 3.2.4: If performance issues arise, seek help from Neo4j community

#### Step 3.3: Test and Refine
**Tech**: pytest  
**Activities**:
- 3.3.1: Write tests for Neo4j queries and compliance rules
- 3.3.2: Run pytest, fix bugs with AI help

### Stage 4: Advanced AI and Polish
**Duration**: 5-6 weeks  
**Goal**: Add GNNs, enterprise features, and deploy  
**Investment**: ~$100-150 (EC2 + Amplify + optional GNN compute)  

#### Step 4.1: Implement Graph Neural Networks
**Tech**: PyTorch, PyTorch Geometric, Neo4j  
**Activities**:
- 4.1.1: Install PyTorch Geometric: `pip install torch-geometric`
- 4.1.2: Export Neo4j graph to PyTorch tensors
- 4.1.3: **[UPDATED]** Define GNN architecture, train on labeled data from RVL-CDIP and CUAD (~90% accuracy goal)
- 4.1.4: Integrate GNN predictions into API/UI
- 4.1.5: If accuracy below target, consider hiring freelancer ($50) for 2-hour GNN optimization

**Sample Code (Enhanced GNN Implementation):**
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE

class InvoiceGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_classes=2):
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

#### Step 4.2: Add Enterprise Features
**Tech**: AWS Cognito, React  
**Activities**:
- 4.2.1: Set up AWS Cognito user pool and identity pool
- 4.2.2: Enhance UI: add export (CSV/PDF), feedback form
- 4.2.3: Add error logging to S3 for audit trail
- 4.2.4: Implement data encryption and security best practices

#### Step 4.3: Deploy and Finalize
**Tech**: AWS ECS, Amplify  
**Activities**:
- 4.3.1: Dockerize Flask API
- 4.3.2: Deploy to ECS (~$0.50/hour), deploy UI to Amplify
- 4.3.3: Test with 100 new invoices from RVL-CDIP, document results
- 4.3.4: Plan v2 features: "ERP integration, SOC 2 compliance"

## AI Assistance Limitations & Mitigations

### 1. Complex System Architecture Integration
**Limitation**: CursorAI struggles with designing optimal data flow between components.  
**Example**: CursorAI might generate code to process invoices sequentially rather than in batches.  
**Mitigation**: Create architecture diagrams manually, use Step Functions for orchestration.

### 2. DeepSeek-VL Optimization
**Limitation**: Fine-tuning a 7B parameter model requires deep ML expertise.  
**Example**: Basic vs. optimized quantization techniques.  
**Mitigation**: Start with baseline code, research specialized repositories.

### 3. Graph Neural Network Implementation
**Limitation**: CursorAI can generate basic GNN code but won't optimize architecture.  
**Example**: Won't automatically suggest graph structure or feature engineering.  
**Mitigation**: Break GNN implementation into tiny steps, test rigorously, consider freelance help.

### 4. Neo4j Query Optimization
**Limitation**: Basic Cypher queries may perform poorly at scale.  
**Example**: Simple queries without proper indexing or constraints.  
**Mitigation**: Research query optimization, post to Neo4j community for help.

### 5. Production-Grade Error Handling
**Limitation**: AI-generated code often lacks comprehensive error handling.  
**Example**: Missing retry logic or transaction management.  
**Mitigation**: Explicitly request robust error handling, test failure scenarios.

## Success Criteria
- **Stage 1**: Successfully extract invoice data from 1,000 PDFs with 90%+ accuracy
- **Stage 2**: Identify duplicate invoices with 85%+ accuracy
- **Stage 3**: Flag compliance issues with 85%+ accuracy
- **Stage 4**: Improve overall system accuracy to 90%+ with GNN implementation

## Risk Management
- **Data Quality Risk**: Mitigate with diverse RVL-CDIP and CUAD datasets
- **Technical Complexity**: Break into small steps, leverage AI assistance strategically
- **Cost Management**: Pause EC2 instances when not in use, monitor usage
- **AI Limitations**: Document challenges, prepare fallback strategies

## Conclusion
This staged implementation approach provides a clear path to building a comprehensive invoice processing system while accounting for the limitations of AI assistance. By breaking down complex tasks and implementing testing at each stage, we can overcome the challenges of sophisticated components like GNNs while maintaining progress toward the enterprise-grade vision. The integration of high-quality datasets like RVL-CDIP and CUAD significantly enhances the project's potential for success. 