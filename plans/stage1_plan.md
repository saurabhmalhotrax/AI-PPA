# Stage 1: Core Extraction and Setup

## Overview
This stage focuses on establishing the foundation of the invoice processing system by setting up the development environment, gathering data sources, implementing basic extraction functionality using DeepSeek-VL, and creating a simple user interface for initial testing.

## Duration & Investment
- **Timeline**: 4-5 weeks
- **Investment**: ~$50-100 (AWS EC2 c6i.large beyond Free Tier if needed)

## Prerequisites
- Basic understanding of Python
- AWS account
- Access to RVL-CDIP and CUAD datasets

## Step-by-Step Implementation Plan

### Step 1.1: Define Scope and Objectives (3-4 days)

#### Tech Stack
- Google Docs
- Miro or Lucidchart (optional for diagrams)

#### Activities
1. **Create Project Documentation**:
   - Create a Google Doc titled "Accounting System Scope"
   - Create a shared folder for all project documentation
   - Set up version control for documents

2. **Define System Capabilities**:
   - Document full system functionality: "System detects duplicates, prevents errors, ensures compliance using DeepSeek-VL, Pinecone, Neo4j, GNNs"
   - List key technical components and their interactions
   - Define user stories and requirements

3. **Establish Stage 1 Goals**:
   - Process a subset of RVL-CDIP invoices
   - Extract key fields (invoice number, amount, vendor, date)
   - Create basic UI for upload and viewing
   - Define success metrics (90% extraction accuracy)

4. **Note Future Vision**:
   - Document future stages (vectors, GNNs, enterprise features)
   - Create a high-level roadmap
   - Identify potential challenges and risks

5. **Review and Finalize**:
   - Review scope document
   - Make necessary adjustments
   - Get final approval and save as baseline

#### Success Criteria
- Comprehensive scope document created
- Clear Stage 1 goals established
- Future vision documented

### Step 1.2: Set Up Development Environment (4-5 days)

#### Tech Stack
- VS Code + CursorAI
- Python 3.9
- Git
- AWS (EC2 c6i.large, S3)

#### Activities
1. **Local Environment Setup**:
   - Install VS Code from https://code.visualstudio.com/
   - Add CursorAI extension
   - Install Python 3.9 using pyenv or official installer
   - Install Git from https://git-scm.com/
   - Set up local project directory

2. **AWS Configuration**:
   - Sign up for AWS account if not already done
   - Set up IAM user with appropriate permissions
   - Configure AWS CLI with credentials
   - Launch c6i.large EC2 instance (4 vCPUs, 8GB RAM, ~$0.34/hour)
   - Create security group with SSH access
   - Create and attach EBS volume for data storage

3. **Create S3 Storage**:
   - Create S3 bucket "invoices-full-system"
   - Configure bucket permissions
   - Set up local AWS CLI for S3 access

4. **EC2 Environment Configuration**:
   - SSH into EC2 instance
   - Update packages: `sudo apt update && sudo apt upgrade -y`
   - Install Python dependencies: `sudo apt install python3-pip python3-venv -y`
   - Set up virtual environment: `python3 -m venv venv`
   - Install Git: `sudo apt install git -y`
   - Clone project repository

5. **Learning Resources**:
   - Watch "Python Basics" tutorial (1 hour): https://www.youtube.com/watch?v=rfscVS0vtbw
   - Watch "AWS EC2 Intro" (30 mins): https://www.youtube.com/watch?v=iHX-jtKIVNA
   - Explore CursorAI capabilities for Python development

#### Success Criteria
- Local development environment configured
- EC2 instance launched and accessible
- S3 bucket created and accessible
- Git repository initialized

### Step 1.3: Gather and Prepare Data (5-6 days)

#### Tech Stack
- Python 3.9
- pandas
- Hugging Face datasets library
- AWS S3

#### Activities
1. **Access RVL-CDIP Dataset**:
   - Install datasets library: `pip install datasets`
   - Load RVL-CDIP dataset with Hugging Face:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("aharley/rvl_cdip")
   ```
   - Filter for invoice class (label 11):
   ```python
   invoices = dataset.filter(lambda example: example['label'] == 11)
   ```
   - Examine dataset structure and sample invoices
   - Extract and save 1,000 invoice images for training

2. **Access CUAD Dataset**:
   - Download CUAD from https://www.atticusprojectai.org/cuad
   - Extract files and explore dataset structure
   - Identify relevant contract clauses for invoice-contract relationships
   - Prepare sample contracts in structured format (CSV/JSON)

3. **Prepare Sample Data**:
   - Select 100 diverse invoices from RVL-CDIP for testing
   - Select 20 representative contracts from CUAD
   - Create metadata CSV files for invoices and contracts

4. **Upload to S3**:
   - Upload prepared data to S3 bucket:
   ```bash
   aws s3 cp ./data/sample_invoices/ s3://invoices-full-system/sample_invoices/ --recursive
   aws s3 cp ./data/sample_contracts/ s3://invoices-full-system/sample_contracts/ --recursive
   ```
   - Verify uploads in AWS console

5. **Create Document Classification Module**:
   - Develop script to classify documents using RVL-CDIP classes
   - Train simple classifier to distinguish invoices from other document types
   - Test classification on mixed document samples
   - Document classification approach and accuracy

#### Success Criteria
- 1,000 invoice images from RVL-CDIP extracted and processed
- CUAD contracts preprocessed and ready for use
- Document classification module implemented
- All sample data uploaded to S3 bucket

### Step 1.4: Extract Data with DeepSeek-VL (7-8 days)

#### Tech Stack
- Python 3.9
- PyTorch
- Transformers
- bitsandbytes
- pdf2image
- AWS EC2 c6i.large

#### Activities
1. **Install Required Packages**:
   - Install base packages:
   ```bash
   pip install torch==2.0.1 transformers==4.30.2 bitsandbytes==0.39.1 pdf2image==1.16.3 boto3==1.28.5 pandas==2.0.3 datasets==2.13.1
   ```
   - Install system dependencies: `sudo apt-get install poppler-utils -y` (for pdf2image)
   - Test imports to verify installation

2. **Set Up DeepSeek-VL Model**:
   - Load and configure DeepSeek-VL 7B with optimized 4-bit quantization:
   ```python
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
   - Verify model loads correctly
   - Test on a single sample image

3. **Develop Extraction Pipeline**:
   - Create script to process invoice images:
   ```python
   def extract_from_invoice(image_path):
       image = Image.open(image_path).convert('RGB')
       prompt = "Extract the following from this invoice: invoice number, date, amount, supplier name, and any notes."
       inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
       
       with torch.no_grad():
           outputs = model.generate(
               **inputs,
               max_new_tokens=512,
               do_sample=False
           )
       
       result = processor.decode(outputs[0], skip_special_tokens=True)
       # Parse structured data from result
       # ...
       return parsed_data
   ```
   - Develop parsing logic to extract structured fields from model output
   - Create batch processing functionality for multiple invoices

4. **Process RVL-CDIP Invoices**:
   - Develop script to process invoices from S3 in batches
   - Process 1,000 invoices from RVL-CDIP dataset
   - Monitor memory usage and processing time
   - Save extraction results as JSON

5. **Convert to Structured Format**:
   - Parse extraction results to CSV format:
   ```python
   import pandas as pd
   
   def json_to_csv(json_path, csv_path):
       with open(json_path, 'r') as f:
           data = json.load(f)
       
       df = pd.DataFrame(data)
       df.to_csv(csv_path, index=False)
   ```
   - Upload CSV to S3
   - Document data format and field definitions

6. **Test and Evaluate Extraction**:
   - Select 50 diverse invoices from RVL-CDIP
   - Perform extraction and manually validate results
   - Calculate accuracy metrics for each field
   - Document common extraction errors
   - Adjust prompts and parsing logic to improve accuracy

#### Success Criteria
- DeepSeek-VL successfully deployed on EC2
- 1,000 RVL-CDIP invoices processed
- Extraction accuracy of 90%+ for key fields
- Structured data saved in accessible format

### Step 1.5: Build Basic UI (7-8 days)

#### Tech Stack
- React
- Node.js
- Flask
- AWS EC2

#### Activities
1. **Set Up React Environment**:
   - Install Node.js locally: `nvm install 18`
   - Create React app: `npx create-react-app invoice-processor-ui`
   - Set up project structure (components, services, etc.)
   - Configure API service for backend communication

2. **Develop UI Components**:
   - Create upload component with drag-and-drop functionality
   - Develop results table for displaying extracted data
   - Add basic validation for file types
   - Implement responsive design using CSS/Tailwind

3. **Set Up Flask Backend**:
   - Install Flask on EC2: `pip install flask flask-cors gunicorn`
   - Create basic API structure:
   ```python
   from flask import Flask, request, jsonify
   from flask_cors import CORS
   import traceback
   import logging
   
   app = Flask(__name__)
   CORS(app)
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
   
   if __name__ == '__main__':
       app.run(debug=True, host='0.0.0.0')
   ```
   - Implement API endpoint for invoice processing
   - Set up environment variables and configuration

4. **Integrate DeepSeek-VL**:
   - Connect Flask API to DeepSeek-VL extraction module
   - Implement file handling and validation
   - Create response formatting

5. **Testing and Debugging**:
   - Test UI locally with mock API
   - Test end-to-end flow with real invoice
   - Validate extraction results
   - Fix bugs and improve UX

6. **Implement Error Handling**:
   - Add comprehensive error handling to Flask API
   - Implement error reporting and logging
   - Create recovery mechanisms for common failure points
   - Test with malformed inputs and server errors
   - Document error codes and recovery procedures

#### Success Criteria
- React UI with file upload functionality
- Flask API successfully connected to DeepSeek-VL
- End-to-end testing completed
- Robust error handling implemented

## Deliverables
1. Project scope document
2. Configured development environment (local and EC2)
3. Prepared datasets (RVL-CDIP invoices and CUAD contracts)
4. Document classification module
5. DeepSeek-VL extraction pipeline
6. Basic UI with upload and results display
7. Technical documentation for all components

## Dependencies and Risks

### Dependencies
- Access to RVL-CDIP and CUAD datasets
- AWS account with appropriate permissions
- Sufficient GPU resources for DeepSeek-VL

### Risks
- DeepSeek-VL may require more resources than c6i.large provides
- Extraction accuracy may be lower than expected on real invoices
- AWS costs may exceed estimates for intensive processing

## Mitigation Strategies
- Test DeepSeek-VL on a small batch before full processing
- Prepare to upgrade to larger instance if needed
- Implement batching to optimize resource usage
- Monitor AWS usage daily to prevent unexpected costs

## Next Steps after Completion
- Review extraction accuracy and identify improvement areas
- Prepare data for vector-based deduplication in Stage 2
- Document lessons learned and technical challenges 