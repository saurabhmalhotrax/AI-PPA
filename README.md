# AI-Powered Invoice Auditor MVP

This project is a quick (30-day) effort to build a smart system that helps with invoice processing. The main goals are:

1.  **Automatically Read Invoices:** Use AI to pull out key information (like invoice number, date, vendor, total amount) from uploaded invoice images or PDFs.
2.  **Find Duplicates:** Cleverly detect if an invoice is a duplicate of one already processed, even if some details are slightly different.
3.  **Basic Compliance Checks:** Show a proof-of-concept for how the system could check if invoices line up with existing contracts (e.g., not exceeding a contract's maximum value).

We're aiming to do this with minimal infrastructure cost, mostly relying on cloud AI services and local tools for this initial version.

## Sprint Breakdown

Here's what we plan to do each "sprint" (a short period of focused work):

### Sprint 0: Getting Ready (0.5 - 1 day)
*   **What we'll do:** Set up the basic project structure. This means creating the code repository (like a project folder with version control), setting up a Python environment (so our code runs consistently), and preparing a `requirements.txt` file (which lists all the software bits our project needs). We'll also get Docker ready for running local services like a database.
*   **Why:** This is like preparing your kitchen and ingredients before you start cooking. It ensures everything is organized and ready for the actual development work.

### Sprint 1: Teaching the AI to Read Invoices (3 - 4 days)
*   **What we'll do:**
    *   Build a "Vision Extractor." This is a piece of code that will connect to an AI service (like OpenAI GPT-4o Vision or Google Gemini 1.5 Pro Vision). We'll tell this AI to "look" at an invoice image and pull out specific details: invoice number, date, vendor name, and the total amount.
    *   Gather sample invoices. We'll use a dataset called **RVL-CDIP**, which is a collection of scanned documents. We'll grab about 200 invoice images from here to test our system.
    *   Process these sample invoices in a batch and save the extracted information into a structured format (like a CSV file).
    *   Write some tests to make sure our Vision Extractor is working correctly.
*   **Significance:**
    *   **Vision Extractor:** This is the core of automated data entry. Instead of someone manually typing in invoice details, the AI does it.
    *   **RVL-CDIP:** This gives us a standard set of real-world (though older) invoices to test and develop our extraction logic without needing sensitive, live data initially.
*   **Why:** This sprint is about turning pictures of invoices into useful, structured data that the computer can understand and work with.

### Sprint 2: Spotting the Doppelgangers (Duplicate Detection) (3 - 4 days)
*   **What we'll do:**
    *   Convert the extracted invoice details into a special text representation.
    *   Use a technique to turn this text into "embeddings" (think of these as numerical fingerprints or signatures for each invoice). We'll use a model called `all-MiniLM-L6-v2` for this.
    *   Use a tool called **FAISS** to store these embeddings and quickly find similar ones. If two invoices have very similar embeddings, they might be duplicates.
    *   Add some smart rules: besides just similarity, we'll check if the amounts are very close and if the dates are within a few days of each other to confirm a duplicate.
    *   Create a small test set of known duplicate and non-duplicate invoices to see how well our system performs.
*   **Significance:**
    *   **Embeddings & FAISS:** This combination allows us to find "semantically" similar invoices, not just exact matches. So, even if an invoice number is typed slightly differently or the date format changes, the system can still flag it as a potential duplicate. FAISS makes this search very fast, even with many invoices.
*   **Why:** This sprint tackles a key pain point: preventing accidental double payments or processing the same invoice multiple times.

### Sprint 3: Connecting the Dots & Basic Rule Checks (2 - 3 days)
*   **What we'll do:**
    *   Load our extracted invoice data and some sample contract data into a **Neo4j graph database**. A graph database is good at showing relationships (e.g., this invoice was issued by this vendor, and this vendor has this contract).
    *   Write a simple rule in Neo4j to check, for example, if an invoice's total amount is greater than the maximum value specified in the vendor's contract.
    *   Set up a tiny "Graph Neural Network" (GNN) demo. This is a more advanced AI technique that can learn from the relationships in the graph. For the MVP, we'll just show it can be trained on a small dataset to potentially identify invoices that might violate *any* contract rule.
*   **Significance:**
    *   **Neo4j (Graph Database):** Helps us see and query complex relationships between invoices, vendors, and contracts, which is hard with traditional tables.
    *   **GNN Demo:** This is a forward-looking piece to show that the system has the potential to learn more complex compliance patterns beyond simple, hardcoded rules.
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

*   **Vision LLM (OpenAI/Gemini):** The AI "eyes" that read the invoice documents.
*   **Sentence-Transformers & FAISS:** Tools to create "fingerprints" (embeddings) of invoices and then quickly find similar fingerprints to detect duplicates.
*   **Neo4j:** A special database that's good at storing and searching for relationships between data points (like invoices, vendors, contracts).
*   **PyG (PyTorch Geometric) / GNN:** A library for building "Graph Neural Networks," which are AIs that can learn from data that's connected in a graph (like our invoice-vendor-contract network).
*   **Streamlit:** A simple way to build basic web applications in Python, perfect for demos.

## What's Next (After this MVP)

If this MVP is successful, the plan is to:
*   Handle many more invoices.
*   Automate the understanding of more complex contract clauses.
*   Add proper user accounts and security.
*   Make the system more robust and deploy it to the cloud.
*   Improve the accuracy of the AI that reads invoices.

---

This README provides a high-level overview. The detailed plan with specific tasks and deliverables for each sprint is in the `plans/MVP plan.md` document.
