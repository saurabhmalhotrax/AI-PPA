# Stage 2: Vector-Based Deduplication

## Overview
This stage builds upon the foundational work of Stage 1 by implementing vector-based deduplication using Pinecone vector database. We will enhance the data extraction process, create embeddings of invoice data, and develop algorithms to detect duplicate invoices across the system.

## Duration & Investment
- **Timeline**: 4-5 weeks
- **Investment**: ~$50-100 (EC2 + Pinecone paid tier if Free Tier exceeded)

## Prerequisites
- Successful completion of Stage 1
- Extracted invoice data from RVL-CDIP dataset
- Functioning DeepSeek-VL extraction pipeline
- Basic understanding of vector embeddings and similarity search

## Step-by-Step Implementation Plan

### Step 2.1: Enhance Data Extraction (6-7 days)

#### Tech Stack
- Python 3.9
- PyTorch
- spaCy
- DeepSeek-VL from Stage 1
- AWS EC2

#### Activities
1. **Set Up NLP Environment**:
   - Install spaCy: `pip install spacy==3.5.3`
   - Download English language model: `python -m spacy download en_core_web_sm`
   - Verify installation: 
   ```python
   import spacy
   nlp = spacy.load("en_core_web_sm")
   doc = nlp("This is a test sentence.")
   print([(token.text, token.pos_) for token in doc])
   ```

2. **Refine DeepSeek-VL Extraction**:
   - Review extraction results from Stage 1
   - Identify common extraction errors and patterns
   - Optimize prompts for improved extraction:
   ```python
   def improved_prompt(image_type):
       prompts = {
           "standard": "Extract these exact fields from this invoice: 1. Invoice Number, 2. Date (in MM/DD/YYYY format), 3. Total Amount (with currency), 4. Vendor/Supplier Name, 5. Any payment terms or notes. Format as JSON.",
           "complex": "This invoice has a complex layout. Carefully extract: 1. Invoice Number (usually top right), 2. Date (find any dates and identify the invoice date), 3. Total Amount (look for 'Total' or largest currency value), 4. Vendor Name (usually at top/header), 5. Payment Terms. Format as JSON."
       }
       return prompts.get(image_type, prompts["standard"])
   ```
   - Create image classification logic to choose appropriate prompts

3. **Implement Post-Processing with spaCy**:
   - Develop named entity recognition for vendor names:
   ```python
   def extract_vendor(text):
       doc = nlp(text)
       orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
       return orgs[0] if orgs else None
   ```
   - Create date normalization function:
   ```python
   def normalize_date(date_str):
       # Convert various date formats to ISO
       # ...
       return normalized_date
   ```
   - Implement amount extraction with regex and validation:
   ```python
   import re
   
   def extract_amount(text):
       # Match currency patterns with regex
       amount_pattern = r'[\$£€]?[\d,]+\.?\d*'
       matches = re.findall(amount_pattern, text)
       # Process and validate matches
       # ...
       return validated_amount
   ```
   - Create quality assurance checks for extracted fields

4. **Process RVL-CDIP Invoices with Enhanced Pipeline**:
   - Update extraction pipeline with new components
   - Reprocess 1,000 RVL-CDIP invoices
   - Save enhanced extraction results as JSON
   - Compare accuracy with Stage 1 results

5. **Create Training Data for Duplicate Detection**:
   - Select 100 invoices from RVL-CDIP
   - Create synthetic duplicates with minor variations:
     - Change date format
     - Alter spacing in vendor name
     - Minor amount differences
   - Label pairs as duplicate/non-duplicate in CSV:
   ```
   invoice_id_1,invoice_id_2,is_duplicate
   inv001,inv001a,1
   inv001,inv002,0
   ```
   - Split into training (80%) and validation (20%) sets

#### Success Criteria
- spaCy pipeline integrated with DeepSeek-VL extraction
- Extraction accuracy improved by at least 10% compared to Stage 1
- 100 labeled invoice pairs (duplicate/non-duplicate) created
- Enhanced extraction pipeline documented

### Step 2.2: Set Up Vector Database (7-8 days)

#### Tech Stack
- Python 3.9
- Pinecone
- SentenceTransformers
- pandas
- AWS EC2

#### Activities
1. **Set Up Pinecone Account**:
   - Sign up for Pinecone at https://www.pinecone.io/
   - Create a starter or standard tier project (depends on vector count)
   - Note API key and environment details
   - Install Pinecone client: `pip install pinecone-client==2.2.1`

2. **Install Embedding Tools**:
   - Install SentenceTransformers: `pip install sentence-transformers==2.2.2`
   - Test embedding generation:
   ```python
   from sentence_transformers import SentenceTransformer
   
   model = SentenceTransformer('all-MiniLM-L6-v2')
   embedding = model.encode("This is a test sentence")
   print(f"Embedding dimension: {len(embedding)}")
   ```

3. **Design Invoice Embedding Strategy**:
   - Develop function to convert invoice data to text representation:
   ```python
   def invoice_to_text(invoice_data):
       """Convert invoice data to a single text string for embedding"""
       return f"Invoice {invoice_data['invoice_number']} from {invoice_data['vendor']} for {invoice_data['amount']} dated {invoice_data['date']}"
   ```
   - Experiment with different text representations
   - Test embedding quality on sample invoices
   - Document chosen embedding approach

4. **Create Pinecone Index**:
   - Initialize Pinecone client:
   ```python
   import pinecone
   import os
   
   # Initialize connection
   pinecone.init(
       api_key=os.environ.get('PINECONE_API_KEY'),
       environment=os.environ.get('PINECONE_ENVIRONMENT')
   )
   
   # Create index if not exists
   if "invoice-index" not in pinecone.list_indexes():
       pinecone.create_index(
           name="invoice-index",
           dimension=384,  # Dimension for all-MiniLM-L6-v2
           metric="cosine"
       )
   
   # Connect to index
   index = pinecone.Index("invoice-index")
   ```
   - Configure index settings (dimensions based on embedding model)
   - Set up environment variables for credentials

5. **Develop Embedding and Storage Script**:
   - Create function to embed and store invoices:
   ```python
   def embed_and_store(invoice_data, index):
       """Embed invoice text and store in Pinecone"""
       try:
           # Convert invoice to text
           invoice_text = invoice_to_text(invoice_data)
           
           # Generate embedding
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
   - Add batch processing capability for efficiency
   - Implement error handling and retry logic

6. **Test Vector Storage with Sample Data**:
   - Test upsert with 10 invoices
   - Verify data stored correctly
   - Execute simple similarity search:
   ```python
   # Search for similar invoices
   results = index.query(
       vector=embedding,
       top_k=5,
       include_metadata=True
   )
   ```
   - Document query response structure

7. **Process Full Invoice Dataset**:
   - Embed and store all 1,000 processed invoices
   - Monitor progress and handle failures
   - Record statistics (processing time, success rate)
   - Test retrieval of random samples

#### Success Criteria
- Pinecone index created and configured
- Embedding strategy documented and tested
- 100 test invoices successfully embedded and stored
- Batch processing script operational
- Basic similarity search functioning

### Step 2.3: Implement Duplicate Detection (8-10 days)

#### Tech Stack
- Python 3.9
- Pinecone
- scikit-learn
- pandas
- Flask (from Stage 1)
- React (from Stage 1)
- AWS EC2

#### Activities
1. **Develop Similarity Calculation Functions**:
   - Create function to search for similar invoices:
   ```python
   def find_similar_invoices(invoice_data, index, threshold=0.9, top_k=10):
       """Find invoices similar to the given invoice"""
       # Convert to text and generate embedding
       invoice_text = invoice_to_text(invoice_data)
       embedding = model.encode(invoice_text).tolist()
       
       # Query Pinecone
       results = index.query(
           vector=embedding,
           top_k=top_k,
           include_metadata=True
       )
       
       # Filter by similarity threshold
       similar_invoices = [
           {
               'id': match['id'],
               'score': match['score'],
               'metadata': match['metadata']
           }
           for match in results['matches']
           if match['score'] >= threshold and match['id'] != invoice_data['invoice_number']
       ]
       
       return similar_invoices
   ```
   - Create function to evaluate vendor name similarity using fuzzy matching:
   ```python
   from fuzzywuzzy import fuzz
   
   def vendor_name_similarity(name1, name2, threshold=80):
       """Calculate similarity between vendor names using fuzzy matching"""
       score = fuzz.token_sort_ratio(name1.lower(), name2.lower())
       return score >= threshold
   ```
   - Add functions for amount and date comparison

2. **Evaluate Similarity Threshold**:
   - Test various similarity thresholds on labeled data
   - Calculate precision and recall for each threshold
   - Visualize ROC curve to select optimal threshold:
   ```python
   import matplotlib.pyplot as plt
   from sklearn.metrics import roc_curve, auc
   
   # Calculate scores for test pairs
   scores = []
   true_labels = []
   
   for _, row in test_pairs.iterrows():
       # Get embeddings for both invoices
       embedding1 = get_invoice_embedding(row['invoice_id_1'])
       embedding2 = get_invoice_embedding(row['invoice_id_2'])
       
       # Calculate cosine similarity
       similarity = calculate_similarity(embedding1, embedding2)
       
       scores.append(similarity)
       true_labels.append(row['is_duplicate'])
   
   # Calculate ROC curve
   fpr, tpr, thresholds = roc_curve(true_labels, scores)
   roc_auc = auc(fpr, tpr)
   
   # Plot ROC curve
   plt.figure()
   plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic')
   plt.legend(loc="lower right")
   plt.savefig('roc_curve.png')
   ```
   - Document selected threshold and rationale

3. **Enhance Duplicate Detection with Machine Learning** (if needed):
   - Prepare feature engineering:
   ```python
   def extract_features(invoice_pair):
       """Extract features from a pair of invoices for ML model"""
       invoice1, invoice2 = invoice_pair
       
       # Vector similarity
       embedding1 = get_invoice_embedding(invoice1['invoice_number'])
       embedding2 = get_invoice_embedding(invoice2['invoice_number'])
       vector_similarity = calculate_similarity(embedding1, embedding2)
       
       # Vendor name similarity
       vendor_similarity = fuzz.token_sort_ratio(
           invoice1['vendor'].lower(), 
           invoice2['vendor'].lower()
       ) / 100.0
       
       # Amount difference (normalized)
       amount_diff = abs(float(invoice1['amount']) - float(invoice2['amount'])) / max(float(invoice1['amount']), 1.0)
       
       # Date difference in days
       date1 = datetime.strptime(invoice1['date'], '%Y-%m-%d')
       date2 = datetime.strptime(invoice2['date'], '%Y-%m-%d')
       date_diff = abs((date1 - date2).days)
       
       return [vector_similarity, vendor_similarity, amount_diff, date_diff]
   ```
   - Train logistic regression model:
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split
   
   # Extract features from labeled pairs
   X = [extract_features(get_invoice_pair(row)) for _, row in labeled_pairs.iterrows()]
   y = labeled_pairs['is_duplicate'].values
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   # Train model
   model = LogisticRegression()
   model.fit(X_train, y_train)
   
   # Evaluate
   y_pred = model.predict(X_test)
   accuracy = (y_pred == y_test).mean()
   print(f"Model accuracy: {accuracy:.2f}")
   ```
   - Save model and preprocessing pipeline

4. **Integrate with Flask API**:
   - Add endpoint for duplicate detection:
   ```python
   @app.route('/api/detect-duplicates', methods=['POST'])
   def detect_duplicates():
       try:
           # Get invoice data from request
           invoice_data = request.json
           
           # Find similar invoices
           similar_invoices = find_similar_invoices(
               invoice_data, 
               index, 
               threshold=0.85
           )
           
           return jsonify({
               "similar_invoices": similar_invoices,
               "count": len(similar_invoices)
           })
       except Exception as e:
           error_id = log_error(e, traceback.format_exc())
           return jsonify({"error": "Duplicate detection failed", "error_id": error_id}), 500
   ```
   - Test API endpoint with sample invoices
   - Set up caching for performance optimization

5. **Update React UI**:
   - Add component to display potential duplicates:
   ```jsx
   const DuplicateWarning = ({ similarInvoices }) => {
     if (!similarInvoices || similarInvoices.length === 0) return null;
     
     return (
       <div className="bg-yellow-100 border-l-4 border-yellow-500 p-4 mb-4">
         <div className="flex">
           <div className="flex-shrink-0">
             <ExclamationIcon className="h-5 w-5 text-yellow-500" />
           </div>
           <div className="ml-3">
             <p className="text-sm text-yellow-700">
               Found {similarInvoices.length} potential duplicate invoice(s).
             </p>
             <ul className="mt-2 text-sm text-yellow-700">
               {similarInvoices.map(inv => (
                 <li key={inv.id}>
                   Invoice {inv.id} from {inv.metadata.vendor} 
                   ({inv.metadata.date}) - {Math.round(inv.score * 100)}% similar
                 </li>
               ))}
             </ul>
           </div>
         </div>
       </div>
     );
   };
   ```
   - Integrate with existing upload flow
   - Add loading states and error handling
   - Test UI with sample invoices

6. **Evaluate System Performance**:
   - Test duplicate detection on 50 pairs from validation set
   - Calculate accuracy, precision, and recall
   - Document performance metrics
   - Identify improvement opportunities

#### Success Criteria
- Duplicate detection accuracy of at least 85% on validation set
- API endpoint for duplicate detection implemented and tested
- UI updated to show potential duplicates
- System handles various invoice formats and variations
- Performance metrics documented

## Deliverables
1. Enhanced extraction pipeline with spaCy integration
2. Pinecone vector database configured and populated
3. Documented embedding strategy and similarity thresholds
4. Duplicate detection algorithm (vector-based or ML-enhanced)
5. API endpoint for duplicate detection
6. Updated UI showing potential duplicates
7. Performance evaluation report

## Dependencies and Risks

### Dependencies
- Successful completion of Stage 1
- Quality of extraction results from DeepSeek-VL
- Pinecone account and API access
- Sufficient labeled data for threshold tuning

### Risks
- Vector similarity alone may not catch all duplicates
- False positives might frustrate users
- Pinecone costs might exceed Free Tier if vector count is high
- Performance degradation with larger invoice sets

## Mitigation Strategies
- Use multi-faceted approach combining vectors with rule-based checks
- Implement confidence scoring to reduce false positives
- Monitor Pinecone usage and optimize index size
- Implement pagination and efficient querying for large datasets

## Next Steps after Completion
- Prepare for Stage 3: Graph and Compliance integration
- Document lessons learned from vector approach
- Collect user feedback on duplicate detection accuracy
- Plan for scaling to larger invoice volumes 