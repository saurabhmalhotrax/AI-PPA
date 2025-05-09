# AI-Powered Invoice Auditor MVP

This project showcases a (30-day) MVP build of an intelligent system designed to revolutionize invoice processing and auditing. It moves beyond simple automation by leveraging advanced AI, graph database technology, and Graph Neural Networks (GNNs) to uncover deeper insights, detect sophisticated anomalies, and streamline compliance.

The main goals for this MVP are:

1.  **Automatically Read Invoices:** Use AI to pull out key information (like invoice number, date, vendor, total amount) from uploaded invoice images or PDFs.
2.  **Find Duplicates:** Intelligently detect if an invoice is a duplicate of one already processed, even if key details (like invoice number or date) are slightly different, using semantic understanding.
3.  **Basic Compliance Checks:** Demonstrate how the system can perform initial checks against contractual agreements (e.g., ensuring an invoice amount doesn't exceed a contract's maximum value) by understanding relationships within the data.
4.  **Uncover Hidden Insights (Proof-of-Concept):** Lay the groundwork for identifying non-obvious risks and relationships using graph technology and GNNs.

We're aiming to do this with minimal infrastructure cost, mostly relying on cloud AI services and local tools for this initial version.

## Why This Approach? The Advantages

This AI-powered auditor offers significant advantages over traditional, often manual or script-based, auditing tools (like those relying solely on relational databases and static dashboards, e.g., PowerBI with custom scripts):

*   **From Reactive to Proactive:** Instead of just reporting what happened (descriptive analytics), this system aims to predict potential issues (predictive analytics) and highlight anomalies you weren't explicitly looking for.
*   **Deep Relational Insight:** Technologies like Neo4j (graph database) allow us to model and query complex relationships between invoices, vendors, contracts, and other entities in a way that's difficult for traditional systems. This helps in visualizing fund flows, identifying indirect connections indicative of collusion, and understanding the full impact of a flagged item.
*   **Sophisticated Anomaly & Fraud Detection:** AI, particularly when combined with GNNs, can learn complex patterns from data and its relationships. This means it can identify subtle, coordinated fraudulent activities or unusual transactions that simple rule-based systems would miss.
*   **Enhanced Accuracy & Efficiency:**
    *   AI-driven data extraction minimizes manual entry errors.
    *   Semantic duplicate detection reduces false positives and negatives.
    *   Graph-based analysis can speed up investigations by making complex scenarios more intuitive to explore.
*   **Scalable Intelligence:** The system is designed to learn and adapt, becoming more effective as it processes more data, unlike static scripts that require manual updates for new scenarios.

## Core Capabilities & Use Cases

*   **Automated & Accurate Invoice Data Extraction:**
    *   **Capability:** Utilizes Vision LLMs (OpenAI GPT-4o/Google Gemini 1.5 Pro) to "read" and extract key information (invoice number, date, vendor, line items, total amount) from various invoice formats (PDFs, images).
    *   **Use Case:** Drastically reduces manual data entry, minimizes errors, and speeds up the initial stages of invoice processing.
*   **Intelligent Duplicate Detection:**
    *   **Capability:** Employs sentence embeddings (`all-MiniLM-L6-v2`) and FAISS for fast, semantic similarity searches, augmented by smart rule checks (e.g., amount proximity, date overlaps).
    *   **Use Case:** Prevents erroneous double payments and duplicate processing by identifying invoices that are substantively the same, even with minor discrepancies.
*   **Graph-Powered Compliance & Contextual Analysis:**
    *   **Capability:** Leverages Neo4j to map relationships between invoices, vendors, and contracts. A GNN demo shows potential for learning complex compliance patterns.
    *   **Use Case:** Performs initial compliance checks (e.g., invoice vs. contract limits). Enables auditors to visually explore and understand the context of any transaction, uncovering non-obvious links or potential policy violations.
*   **Foundation for Advanced Risk Analytics:**
    *   **Capability:** The combination of AI, Neo4j, and GNNs provides a powerful platform for future development of predictive risk scoring and sophisticated pattern recognition.
    *   **Use Case:** Early identification of emerging fraud schemes, high-risk transactions, or systemic compliance vulnerabilities.

## Sprint Breakdown

Here's what we plan to do each "sprint" (a short period of focused work):

### Sprint 0: Getting Ready (0.5 - 1 day)
*   **What we'll do:** Set up the basic project structure. This means creating the code repository (like a project folder with version control), setting up a Python environment (so our code runs consistently), and preparing a `requirements.txt` file (which lists all the software bits our project needs). We'll also get Docker ready for running local services like a database.
*   **Why:** This is like preparing your kitchen and ingredients before you start cooking. It ensures everything is organized and ready for the actual development work.

### Sprint 1: Teaching the AI to Read Invoices (3 - 4 days)
*   **What we'll do:**
    *   Build a "Vision Extractor." This is a piece of code that will connect to an AI service (like OpenAI GPT-4o Vision or Google Gemini 1.5 Pro Vision). We'll tell this AI to "look" at an invoice image and pull out specific details: invoice number, date, vendor name, and the total amount. This forms the basis of our automated data capture.
    *   Gather sample invoices. We'll use a dataset called **RVL-CDIP**, which is a collection of scanned documents. We'll grab about 200 invoice images from here to test our system.
    *   Process these sample invoices in a batch and save the extracted information into a structured format (like a CSV file).
    *   Write some tests to make sure our Vision Extractor is working correctly.
*   **Significance:**
    *   **Vision Extractor:** This is the core of automated data entry. Instead of someone manually typing in invoice details, the AI does it, saving time and reducing errors.
    *   **RVL-CDIP:** This gives us a standard set of real-world (though older) invoices to test and develop our extraction logic without needing sensitive, live data initially.
*   **Why:** This sprint is about turning pictures of invoices into useful, structured data that the computer can understand and work with.

### Sprint 2: Spotting the Doppelgangers (Duplicate Detection) (3 - 4 days)
*   **What we'll do:**
    *   Convert the extracted invoice details into a special text representation.
    *   Use a technique to turn this text into "embeddings" (think of these as numerical fingerprints or signatures for each invoice). We'll use a model called `all-MiniLM-L6-v2` for this, enabling semantic comparison.
    *   Use a tool called **FAISS** to store these embeddings and quickly find similar ones. If two invoices have very similar embeddings, they might be duplicates.
    *   Add some smart rules: besides just similarity, we'll check if the amounts are very close and if the dates are within a few days of each other to confirm a duplicate.
    *   Create a small test set of known duplicate and non-duplicate invoices to see how well our system performs.
*   **Significance:**
    *   **Embeddings & FAISS:** This combination allows us to find "semantically" similar invoices, not just exact matches. So, even if an invoice number is typed slightly differently or the date format changes, the system can still flag it as a potential duplicate. FAISS makes this search very fast, even with many invoices, which is crucial for scalability.
*   **Why:** This sprint tackles a key pain point: preventing accidental double payments or processing the same invoice multiple times.

### Sprint 3: Connecting the Dots & Basic Rule Checks (2 - 3 days)
*   **What we'll do:**
    *   Load our extracted invoice data and some sample contract data into a **Neo4j graph database**. A graph database is good at showing relationships (e.g., this invoice was issued by this vendor, and this vendor has this contract).
    *   Write a simple rule in Neo4j to check, for example, if an invoice's total amount is greater than the maximum value specified in the vendor's contract.
    *   Set up a tiny "Graph Neural Network" (GNN) demo using PyTorch Geometric (PyG). This is a more advanced AI technique that can learn from the relationships in the graph. For the MVP, we'll just show it can be trained on a small dataset to potentially identify invoices that might violate *any* contract rule by learning from relational patterns.
*   **Significance:**
    *   **Neo4j (Graph Database):** Helps us see and query complex relationships between invoices, vendors, and contracts, which is hard with traditional tables. This is key for contextual understanding and advanced analysis.
    *   **GNN Demo:** This is a forward-looking piece to show that the system has the potential to learn more complex compliance patterns and anomalies beyond simple, hardcoded rules, paving the way for predictive capabilities.
*   **Why:** This sprint demonstrates how the system can go beyond duplicate detection and start performing basic compliance checks by understanding the context around an invoice.

### Sprint 4: Show and Tell (The User Interface) (2 - 3 days)
*   **What we'll do:**
    *   Build a very simple web interface using **Streamlit**.
    *   Users will be able to upload an invoice (PDF/image).
    *   The interface will show the data extracted by the Vision API.
    *   It will then report if any potential duplicates are found.
    *   It will also show if the invoice violates the basic contract rule we set up with Neo4j.
    *   (Maybe) It will show the GNN's prediction about compliance.
    *   Prepare a short presentation and video demo of the working system.
*   **Why:** This makes the system tangible. It's one thing to have scripts that run, but a UI allows someone to interact with it and see the results directly.

## Key Technologies (Simplified)

*   **Vision LLM (OpenAI/Gemini):** The AI "eyes" that read the invoice documents. **Benefit:** Automated, accurate data extraction from diverse invoice formats, reducing manual effort and errors.
*   **Sentence-Transformers & FAISS:** Tools to create "fingerprints" (embeddings) of invoices and then quickly find similar fingerprints to detect duplicates. **Benefit:** Intelligent duplicate detection that goes beyond exact matches, identifying subtle similarities and reducing false positives/negatives.
*   **Neo4j:** A special database that's good at storing and searching for relationships between data points (like invoices, vendors, contracts). **Benefit:** Reveals complex relationships and connections within your financial data, enabling deeper insights, impact analysis, and contextual understanding than traditional databases.
*   **PyG (PyTorch Geometric) / GNN:** A library for building "Graph Neural Networks," which are AIs that can learn from data that's connected in a graph (like our invoice-vendor-contract network). **Benefit:** Empowers predictive analytics and the discovery of sophisticated, non-obvious patterns indicative of risk or non-compliance that rule-based systems would miss.
*   **Streamlit:** A simple way to build basic web applications in Python, perfect for demos. **Benefit:** Rapidly demonstrates the core functionalities through an interactive user interface, making the system tangible for stakeholders.

## What's Next (After this MVP)

If this MVP is successful, the vision is to evolve this into an enterprise-grade auditing and financial intelligence platform:

*   **Scale for Volume:** Handle significantly larger volumes of invoices and related financial documents.
*   **Advanced Contract Intelligence:** Automate the understanding and verification of complex contract clauses, terms, and obligations.
*   **Sophisticated GNN-driven Insights:**
    *   Develop robust predictive models for risk scoring (fraud, compliance, operational).
    *   Implement advanced anomaly detection that learns continuously.
    *   Provide explainable AI insights into why certain items are flagged.
*   **Enterprise Integration:** Seamlessly integrate with existing ERP, accounting, and procurement systems.
*   **User Management & Security:** Implement comprehensive user roles, permissions, and robust security protocols.
*   **Enhanced User Experience:** Develop a more sophisticated and customizable user interface with advanced visualization and reporting tools.
*   **Natural Language Querying:** Allow users to interrogate the data and insights using natural language questions.
*   **Continuous Improvement:** Continuously refine the accuracy and capabilities of the AI models based on user feedback and new data.

---

This README provides a high-level overview. The detailed plan with specific tasks and deliverables for each sprint is in the `plans/MVP plan.md` document.
