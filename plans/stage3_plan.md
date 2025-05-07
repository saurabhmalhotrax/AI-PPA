# Stage 3: Graph and Compliance

## Overview
This stage focuses on implementing a graph database to model the relationships between invoices, contracts, vendors, and compliance rules. By leveraging Neo4j and the CUAD dataset, we'll build a powerful system to detect compliance issues and analyze the complex relationships in invoice data.

## Duration & Investment
- **Timeline**: 4-5 weeks
- **Investment**: ~$50-100 (EC2)

## Prerequisites
- Successful completion of Stages 1 and 2
- Processed invoice data from RVL-CDIP dataset
- Preprocessed contract data from CUAD dataset
- Functioning vector-based duplicate detection system

## Step-by-Step Implementation Plan

### Step 3.1: Set Up Memory Graph (8-10 days)

#### Tech Stack
- Neo4j Community Edition
- Python 3.9
- py2neo
- pandas
- AWS EC2

#### Activities
1. **Install Neo4j on EC2**:
   - Update package info: `sudo apt update`
   - Install Java: `sudo apt install openjdk-11-jdk -y`
   - Add Neo4j repository:
   ```bash
   wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
   echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
   sudo apt update
   ```
   - Install Neo4j: `sudo apt install neo4j -y`
   - Start Neo4j service: `sudo systemctl start neo4j`
   - Enable at boot: `sudo systemctl enable neo4j`
   - Configure memory settings:
   ```bash
   sudo nano /etc/neo4j/neo4j.conf
   # Add/modify these lines
   dbms.memory.heap.initial_size=1G
   dbms.memory.heap.max_size=2G
   ```
   - Restart Neo4j: `sudo systemctl restart neo4j`
   - Check status: `sudo systemctl status neo4j`

2. **Secure Neo4j Installation**:
   - Change default password:
   ```bash
   curl -H "Content-Type: application/json" -X POST -d '{"password":"your-new-password"}' -u neo4j:neo4j http://localhost:7474/user/neo4j/password
   ```
   - Update Neo4j configuration for remote access (if needed):
   ```bash
   sudo nano /etc/neo4j/neo4j.conf
   # Uncomment and configure these lines
   dbms.connector.http.listen_address=0.0.0.0:7474
   dbms.connector.bolt.listen_address=0.0.0.0:7687
   ```
   - Configure firewall: `sudo ufw allow 7474,7687/tcp`
   - Restart Neo4j: `sudo systemctl restart neo4j`

3. **Set Up Python Environment for Neo4j**:
   - Install py2neo: `pip install py2neo==2021.2.3`
   - Test connection:
   ```python
   from py2neo import Graph
   graph = Graph("bolt://localhost:7687", auth=("neo4j", "your-new-password"))
   print(graph.run("MATCH (n) RETURN count(n) AS count").data())
   ```
   - Create basic wrapper class:
   ```python
   class GraphDatabase:
       def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="your-new-password"):
           self.graph = Graph(uri, auth=(user, password))
           
       def clear_all(self):
           """Clear all nodes and relationships"""
           self.graph.run("MATCH (n) DETACH DELETE n")
           
       def run_query(self, query, parameters=None):
           """Run a Cypher query with parameters"""
           return self.graph.run(query, parameters or {}).data()
   ```

4. **Define Graph Schema**:
   - Identify entity types: Invoice, Supplier, Contract, Clause
   - Define relationship types: ISSUED_BY, DUPLICATE_OF, LINKED_TO, CONTAINS_CLAUSE
   - Create schema constraints and indexes:
   ```python
   def create_schema(graph):
       """Create schema constraints and indexes"""
       # Unique constraints for nodes
       constraints = [
           "CREATE CONSTRAINT invoice_id IF NOT EXISTS FOR (i:Invoice) REQUIRE i.invoice_id IS UNIQUE",
           "CREATE CONSTRAINT vendor_name IF NOT EXISTS FOR (v:Vendor) REQUIRE v.name IS UNIQUE",
           "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (c:Contract) REQUIRE c.contract_id IS UNIQUE",
           "CREATE CONSTRAINT clause_type IF NOT EXISTS FOR (cl:Clause) REQUIRE cl.type IS UNIQUE"
       ]
       
       # Execute constraints
       for constraint in constraints:
           graph.run(constraint)
       
       # Create indexes for performance
       indexes = [
           "CREATE INDEX invoice_date IF NOT EXISTS FOR (i:Invoice) ON (i.date)",
           "CREATE INDEX invoice_amount IF NOT EXISTS FOR (i:Invoice) ON (i.amount)"
       ]
       
       for index in indexes:
           graph.run(index)
   ```

5. **Map CUAD Clause Categories**:
   - Download and parse CUAD dataset clause categories
   - Identify relevant clause types for invoice-contract relationships:
   ```python
   def load_cuad_clause_categories():
       """Load and process CUAD clause categories"""
       # Important clause types for invoices
       relevant_clauses = [
           "Payment Terms",
           "Limitation of Liability",
           "Termination for Convenience",
           "Insurance",
           "Warranties",
           "Volume Restrictions",
           "Minimum Commitment",
           "Most Favored Nation",
           "Price Restrictions",
           "Material Breach Cure Period", 
           "Audit Rights"
       ]
       
       return relevant_clauses
   ```
   - Create mapping from clause types to Neo4j nodes

6. **Develop Graph Loading Script**:
   - Create node creation functions:
   ```python
   def create_invoice_graph(graph, invoice_df, contracts_df, duplicate_pairs, cuad_clauses):
       """Create graph from invoice data, contracts and known duplicates"""
       # Start a new transaction
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
               value=float(row['value']),
               start_date=row['start_date'],
               end_date=row['end_date']
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
       
       # Create duplicate relationships from Stage 2
       for id1, id2 in duplicate_pairs:
           if id1 in invoices and id2 in invoices:
               duplicate_rel = Relationship(invoices[id1], "DUPLICATE_OF", invoices[id2])
               tx.create(duplicate_rel)
       
       # Link invoices to contracts based on vendor
       for inv_id, invoice in invoices.items():
           vendor_name = invoice['vendor']
           for contract_id, contract in contracts.items():
               if contract['vendor'] == vendor_name:
                   tx.create(Relationship(invoice, "UNDER_CONTRACT", contract))
       
       # Commit all changes
       tx.commit()
       return len(invoices), len(contracts)
   ```
   - Implement data validation and error handling

7. **Process and Load Data**:
   - Prepare invoice data from Stages 1-2
   - Process CUAD contracts for graph loading
   - Execute graph loading script
   - Verify data loaded correctly:
   ```python
   def verify_graph_data(graph):
       """Verify data loaded correctly"""
       counts = {
           "Invoices": graph.run("MATCH (i:Invoice) RETURN count(i) AS count").data()[0]['count'],
           "Vendors": graph.run("MATCH (v:Vendor) RETURN count(v) AS count").data()[0]['count'],
           "Contracts": graph.run("MATCH (c:Contract) RETURN count(c) AS count").data()[0]['count'],
           "Clauses": graph.run("MATCH (cl:Clause) RETURN count(cl) AS count").data()[0]['count'],
           "ISSUED_BY": graph.run("MATCH ()-[r:ISSUED_BY]->() RETURN count(r) AS count").data()[0]['count'],
           "DUPLICATE_OF": graph.run("MATCH ()-[r:DUPLICATE_OF]->() RETURN count(r) AS count").data()[0]['count'],
           "UNDER_CONTRACT": graph.run("MATCH ()-[r:UNDER_CONTRACT]->() RETURN count(r) AS count").data()[0]['count'],
           "CONTAINS_CLAUSE": graph.run("MATCH ()-[r:CONTAINS_CLAUSE]->() RETURN count(r) AS count").data()[0]['count']
       }
       
       return counts
   ```

8. **Test Basic Graph Queries**:
   - Test simple relationship queries:
   ```python
   # Find duplicate invoices
   graph.run("MATCH (i:Invoice)-[:DUPLICATE_OF]->(j:Invoice) RETURN i.invoice_id, j.invoice_id")
   
   # Find invoices without contracts
   graph.run("""
   MATCH (i:Invoice)-[:ISSUED_BY]->(v:Vendor)
   WHERE NOT EXISTS {
       MATCH (i)-[:UNDER_CONTRACT]->(:Contract)
   }
   RETURN i.invoice_id, i.amount, v.name
   """)
   
   # Find vendors with multiple contracts
   graph.run("""
   MATCH (v:Vendor)<-[:WITH_VENDOR]-(c:Contract)
   WITH v, count(c) AS contract_count
   WHERE contract_count > 1
   RETURN v.name, contract_count
   ORDER BY contract_count DESC
   """)
   ```
   - Document query results and performance

9. **Identify and Fix Query Performance Issues**:
   - Profile slow queries:
   ```cypher
   PROFILE MATCH (i:Invoice)-[:ISSUED_BY]->(v:Vendor)
   WHERE i.amount > 1000
   RETURN i, v
   LIMIT 100
   ```
   - Add additional indexes based on query patterns
   - Optimize complex traversals
   - Document performance improvements

#### Success Criteria
- Neo4j successfully installed and secured on EC2
- Graph schema defined with appropriate constraints and indexes
- CUAD clause categories mapped to graph schema
- Invoice and contract data successfully loaded
- Basic queries performing efficiently (< 1 second response time)
- Documentation of graph structure and query patterns

### Step 3.2: Add Compliance Checker (6-8 days)

#### Tech Stack
- Python 3.9
- Neo4j
- py2neo
- pandas
- AWS EC2

#### Activities
1. **Define Compliance Rules**:
   - Analyze CUAD clause categories for compliance rules
   - Define detection logic for common issues:
   ```python
   def define_compliance_rules():
       """Define rules for invoice compliance checking"""
       rules = [
           {
               "name": "invoice_exceeds_contract_value",
               "description": "Invoice amount exceeds contract value",
               "query": """
               MATCH (i:Invoice)-[:UNDER_CONTRACT]->(c:Contract)
               WHERE i.amount > c.value
               RETURN i.invoice_id AS invoice_id, i.amount AS invoice_amount, 
                      c.contract_id AS contract_id, c.value AS contract_value
               """
           },
           {
               "name": "invoice_outside_contract_period",
               "description": "Invoice date outside contract period",
               "query": """
               MATCH (i:Invoice)-[:UNDER_CONTRACT]->(c:Contract)
               WHERE date(i.date) < date(c.start_date) OR date(i.date) > date(c.end_date)
               RETURN i.invoice_id AS invoice_id, i.date AS invoice_date,
                      c.contract_id AS contract_id, c.start_date AS start_date, c.end_date AS end_date
               """
           },
           {
               "name": "multiple_invoices_same_period",
               "description": "Multiple invoices from same vendor in short period",
               "query": """
               MATCH (i1:Invoice)-[:ISSUED_BY]->(v:Vendor)<-[:ISSUED_BY]-(i2:Invoice)
               WHERE i1.invoice_id < i2.invoice_id
                 AND abs(duration.between(date(i1.date), date(i2.date)).days) < 7
                 AND NOT (i1)-[:DUPLICATE_OF]-(i2)
               RETURN i1.invoice_id AS invoice1, i2.invoice_id AS invoice2,
                      i1.date AS date1, i2.date AS date2, v.name AS vendor
               """
           },
           {
               "name": "missing_payment_terms",
               "description": "Invoice linked to contract without payment terms clause",
               "query": """
               MATCH (i:Invoice)-[:UNDER_CONTRACT]->(c:Contract)
               WHERE NOT EXISTS {
                   MATCH (c)-[:CONTAINS_CLAUSE]->(:Clause {type: 'Payment Terms'})
               }
               RETURN i.invoice_id AS invoice_id, c.contract_id AS contract_id
               """
           }
       ]
       return rules
   ```

2. **Implement Compliance Checking Engine**:
   - Create compliance checking class:
   ```python
   class ComplianceChecker:
       def __init__(self, graph):
           self.graph = graph
           self.rules = define_compliance_rules()
           
       def check_all_rules(self):
           """Run all compliance checks and return issues"""
           all_issues = {}
           
           for rule in self.rules:
               rule_issues = self.check_rule(rule)
               if rule_issues:
                   all_issues[rule['name']] = {
                       'description': rule['description'],
                       'issues': rule_issues
                   }
                   
           return all_issues
       
       def check_rule(self, rule):
           """Run a specific compliance rule and return issues"""
           return self.graph.run(rule['query']).data()
       
       def check_invoice(self, invoice_id):
           """Check compliance for a specific invoice"""
           invoice_issues = {}
           
           for rule in self.rules:
               # Modify query to filter for specific invoice
               invoice_query = rule['query'].replace(
                   "MATCH (i", f"MATCH (i:Invoice {{invoice_id: '{invoice_id}'}}")
               
               rule_issues = self.graph.run(invoice_query).data()
               if rule_issues:
                   invoice_issues[rule['name']] = {
                       'description': rule['description'],
                       'issues': rule_issues
                   }
                   
           return invoice_issues
   ```
   - Implement rule-specific validation functions

3. **Test Compliance Rules on Sample Data**:
   - Create test dataset with intentional compliance issues
   - Run compliance checker on test data
   - Verify all rules are detecting issues correctly
   - Document test results and detection rates

4. **Query Neo4j for Contract Links**:
   - Develop queries to link invoices to relevant contracts:
   ```python
   def find_relevant_contracts(graph, invoice_data):
       """Find contracts relevant to an invoice"""
       query = """
       MATCH (v:Vendor {name: $vendor_name})
       MATCH (v)<-[:WITH_VENDOR]-(c:Contract)
       WHERE date($invoice_date) >= date(c.start_date) AND 
             date($invoice_date) <= date(c.end_date)
       RETURN c.contract_id AS contract_id, c.value AS value,
              c.start_date AS start_date, c.end_date AS end_date
       """
       
       return graph.run(
           query, 
           vendor_name=invoice_data['vendor'],
           invoice_date=invoice_data['date']
       ).data()
   ```
   - Test on 50 invoices from RVL-CDIP
   - Record linking accuracy and performance

5. **Implement Compliance Reporting**:
   - Create reporting functions for compliance issues:
   ```python
   def generate_compliance_report(issues):
       """Generate a formatted compliance report from issues"""
       report = {}
       
       # Count issues by type
       report['issue_counts'] = {name: len(data['issues']) for name, data in issues.items()}
       report['total_issues'] = sum(report['issue_counts'].values())
       
       # Top vendors with issues
       vendors_with_issues = {}
       for rule_name, rule_data in issues.items():
           for issue in rule_data['issues']:
               if 'vendor' in issue:
                   vendors_with_issues[issue['vendor']] = vendors_with_issues.get(issue['vendor'], 0) + 1
       
       report['top_vendors'] = sorted(
           [{'vendor': k, 'issues': v} for k, v in vendors_with_issues.items()],
           key=lambda x: x['issues'],
           reverse=True
       )[:10]
       
       # Sample issues
       report['samples'] = {name: data['issues'][:5] for name, data in issues.items()}
       
       return report
   ```
   - Create visualizations of compliance issues
   - Develop scheduled reporting functionality

6. **Update Flask API with Compliance Endpoints**:
   - Add endpoint for checking invoice compliance:
   ```python
   @app.route('/api/check-compliance', methods=['POST'])
   def check_compliance():
       try:
           # Get invoice data from request
           invoice_data = request.json
           
           # Find or create invoice node
           # (You might update existing or create temporary)
           
           # Check compliance
           checker = ComplianceChecker(graph)
           issues = checker.check_invoice(invoice_data['invoice_number'])
           
           return jsonify({
               "invoice_id": invoice_data['invoice_number'],
               "compliance_issues": issues,
               "has_issues": len(issues) > 0
           })
       except Exception as e:
           error_id = log_error(e, traceback.format_exc())
           return jsonify({"error": "Compliance check failed", "error_id": error_id}), 500
   ```
   - Add endpoint for generating compliance reports:
   ```python
   @app.route('/api/compliance-report', methods=['GET'])
   def compliance_report():
       try:
           # Get all compliance issues
           checker = ComplianceChecker(graph)
           all_issues = checker.check_all_rules()
           
           # Generate report
           report = generate_compliance_report(all_issues)
           
           return jsonify(report)
       except Exception as e:
           error_id = log_error(e, traceback.format_exc())
           return jsonify({"error": "Report generation failed", "error_id": error_id}), 500
   ```
   - Test API endpoints with sample data

7. **Update React UI with Compliance Alerts**:
   - Add compliance warning component:
   ```jsx
   const ComplianceWarning = ({ complianceIssues }) => {
     if (!complianceIssues || Object.keys(complianceIssues).length === 0) return null;
     
     return (
       <div className="bg-red-100 border-l-4 border-red-500 p-4 mb-4">
         <div className="flex">
           <div className="flex-shrink-0">
             <ExclamationIcon className="h-5 w-5 text-red-500" />
           </div>
           <div className="ml-3">
             <p className="text-sm text-red-700">
               Found {Object.keys(complianceIssues).length} compliance issue(s).
             </p>
             <ul className="mt-2 text-sm text-red-700">
               {Object.entries(complianceIssues).map(([ruleName, data]) => (
                 <li key={ruleName}>
                   {data.description}
                 </li>
               ))}
             </ul>
           </div>
         </div>
       </div>
     );
   };
   ```
   - Integrate with existing invoice processing UI
   - Add compliance report page with charts and tables
   - Test UI with sample compliance issues

#### Success Criteria
- At least 4 compliance rules implemented and tested
- Compliance checking with Neo4j queries running efficiently
- API endpoints for compliance checking implemented
- UI updated to show compliance warnings
- Documentation of compliance rules and detection methods

### Step 3.3: Test and Refine (5-6 days)

#### Tech Stack
- Python 3.9
- pytest
- Neo4j
- Flask
- React
- AWS EC2

#### Activities
1. **Set Up Testing Framework**:
   - Install pytest: `pip install pytest==7.3.1 pytest-mock==3.10.0`
   - Create test directory structure:
   ```
   tests/
   ├── __init__.py
   ├── conftest.py
   ├── test_graph_schema.py
   ├── test_compliance_rules.py
   ├── test_api_endpoints.py
   └── test_data/
       ├── sample_invoices.json
       ├── sample_contracts.json
       └── expected_results.json
   ```
   - Add testing utilities in conftest.py:
   ```python
   import pytest
   from py2neo import Graph
   import json
   import os
   
   @pytest.fixture
   def test_graph():
       """Create a test graph with sample data"""
       graph = Graph("bolt://localhost:7687", auth=("neo4j", "your-password"))
       
       # Clear existing data
       graph.run("MATCH (n) DETACH DELETE n")
       
       # Load test schema
       # ...
       
       # Load test data
       # ...
       
       return graph
   
   @pytest.fixture
   def sample_invoices():
       """Load sample invoice data"""
       with open("tests/test_data/sample_invoices.json") as f:
           return json.load(f)
   
   @pytest.fixture
   def sample_contracts():
       """Load sample contract data"""
       with open("tests/test_data/sample_contracts.json") as f:
           return json.load(f)
   ```

2. **Write Tests for Graph Schema**:
   - Create tests to verify schema constraints and indexes:
   ```python
   def test_schema_constraints(test_graph):
       """Test that schema constraints are created correctly"""
       constraints = test_graph.run("SHOW CONSTRAINTS").data()
       
       # Check that all required constraints exist
       constraint_names = [c['name'] for c in constraints]
       assert "invoice_id" in constraint_names
       assert "vendor_name" in constraint_names
       assert "contract_id" in constraint_names
       assert "clause_type" in constraint_names
   
   def test_schema_indexes(test_graph):
       """Test that indexes are created correctly"""
       indexes = test_graph.run("SHOW INDEXES").data()
       
       # Check that all required indexes exist
       index_names = [i['name'] for i in indexes]
       assert "invoice_date" in index_names
       assert "invoice_amount" in index_names
   ```
   - Test data loading functions

3. **Write Tests for Compliance Rules**:
   - Create tests for each compliance rule:
   ```python
   def test_invoice_exceeds_contract_value(test_graph, sample_invoices, sample_contracts):
       """Test detection of invoices exceeding contract value"""
       from compliance_checker import ComplianceChecker
       
       # Create test scenario with invoice > contract value
       # ...
       
       # Check compliance
       checker = ComplianceChecker(test_graph)
       issues = checker.check_rule({
           "name": "invoice_exceeds_contract_value",
           "description": "Invoice amount exceeds contract value",
           "query": """
           MATCH (i:Invoice)-[:UNDER_CONTRACT]->(c:Contract)
           WHERE i.amount > c.value
           RETURN i.invoice_id AS invoice_id, i.amount AS invoice_amount, 
                  c.contract_id AS contract_id, c.value AS contract_value
           """
       })
       
       # Verify issue detected
       assert len(issues) == 1
       assert issues[0]['invoice_id'] == 'test-invoice-001'
       assert issues[0]['invoice_amount'] > issues[0]['contract_value']
   ```
   - Test edge cases and false positives

4. **Write Tests for API Endpoints**:
   - Create tests for Flask API endpoints:
   ```python
   def test_compliance_check_endpoint(client, sample_invoices):
       """Test the compliance check endpoint"""
       response = client.post(
           '/api/check-compliance',
           json=sample_invoices[0]
       )
       
       assert response.status_code == 200
       data = response.json
       
       # Verify response structure
       assert 'invoice_id' in data
       assert 'compliance_issues' in data
       assert 'has_issues' in data
       
       # Verify specific issues if expected
       if data['has_issues']:
           assert 'invoice_exceeds_contract_value' in data['compliance_issues']
   ```
   - Test error handling and edge cases

5. **Run Tests and Fix Bugs**:
   - Execute test suite: `python -m pytest tests/`
   - Address failures and bugs
   - Document fixes and improvements
   - Rerun tests to verify fixes

6. **Performance Testing**:
   - Test system with larger dataset (500+ invoices)
   - Measure and optimize query performance
   - Identify and address bottlenecks
   - Document performance characteristics

7. **Usability Testing**:
   - Test end-to-end workflows
   - Verify UI feedback for compliance issues
   - Test edge cases and error scenarios
   - Document usability improvements

#### Success Criteria
- Comprehensive test suite covering graph schema, compliance rules, and API endpoints
- All tests passing with 90%+ coverage
- System handles 500+ invoices with acceptable performance
- Documentation of tests, bugs fixed, and performance optimizations

## Deliverables
1. Neo4j graph database with invoice, contract, and compliance data
2. Graph schema documentation and visualization
3. Compliance rules implementation with Neo4j queries
4. API endpoints for compliance checking and reporting
5. Updated UI with compliance alerts
6. Test suite for graph schema, compliance rules, and API endpoints
7. Performance optimization documentation

## Dependencies and Risks

### Dependencies
- Successful completion of Stages 1 and 2
- Quality of invoice data extraction from DeepSeek-VL
- Availability of CUAD contract data
- Neo4j performance on EC2 instance

### Risks
- Graph queries may become complex and slow with larger datasets
- Contract data from CUAD may require extensive preprocessing
- Linking invoices to contracts accurately may be challenging
- False positives in compliance checks may frustrate users

## Mitigation Strategies
- Optimize Neo4j queries with proper indexing and query planning
- Create preprocessing pipeline for CUAD contract data
- Implement fuzzy matching for invoice-contract linking
- Add confidence scores to compliance issues to reduce false positives
- Use pagination and efficient querying for large datasets

## Next Steps after Completion
- Prepare for Stage 4: Advanced AI and Polish
- Plan GNN implementation to enhance graph analysis
- Document lessons learned from graph and compliance approach
- Collect feedback on compliance detection accuracy 