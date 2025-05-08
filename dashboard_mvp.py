import streamlit as st
import pandas as pd
import numpy as np # For dummy data

# Load the invoice data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Basic cleaning: Convert 'total_amount' to numeric, coercing errors
        if 'total_amount' in df.columns:
            df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        # Convert 'date' to datetime, coercing errors
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

invoice_df = load_data("data/extracted_invoices.csv")

st.set_page_config(layout="wide")
st.title("Invoice Analysis & Compliance Dashboard MVP")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Invoice Data Explorer",
    "🕸️ Neo4j Graph Visualizer (Simulated)",
    "🛡️ GNN Compliance Dashboard (Simulated)",
    "❓ Ad-hoc Query Simulation"
])

with tab1:
    st.header("Invoice Data Explorer")
    if not invoice_df.empty:
        st.write("Browse and filter your extracted invoice data.")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            vendors = ["All"] + sorted(invoice_df['vendor'].astype(str).unique().tolist())
            selected_vendor = st.selectbox("Filter by Vendor", vendors)
        with col2:
            min_date = invoice_df['date'].min()
            max_date = invoice_df['date'].max()
            if pd.isna(min_date) or pd.isna(max_date):
                st.warning("Date data is incomplete for range filtering.")
                selected_date_range = None
            else:
                selected_date_range = st.date_input("Filter by Date Range",
                                                    value=(min_date, max_date),
                                                    min_value=min_date,
                                                    max_value=max_date,
                                                    key="invoice_date_filter")
        with col3:
            # Ensure total_amount is numeric for filtering
            if pd.api.types.is_numeric_dtype(invoice_df['total_amount']):
                min_amount = invoice_df['total_amount'].min()
                max_amount = invoice_df['total_amount'].max()
                if pd.isna(min_amount) or pd.isna(max_amount):
                     selected_amount_range = st.slider("Filter by Total Amount", 0.0, 100000.0, (0.0, 100000.0))
                else:
                    selected_amount_range = st.slider("Filter by Total Amount", float(min_amount), float(max_amount), (float(min_amount), float(max_amount)))
            else:
                selected_amount_range = None # Explicitly set to None if not numeric
                st.warning("Total amount column is not numeric and cannot be used for range slider.")


        # Apply filters
        filtered_df = invoice_df.copy()
        if selected_vendor != "All":
            filtered_df = filtered_df[filtered_df['vendor'] == selected_vendor]

        if selected_date_range and len(selected_date_range) == 2 and not pd.isna(selected_date_range[0]) and not pd.isna(selected_date_range[1]):
            start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
            # Ensure 'date' column in filtered_df is also datetime for comparison
            if pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
                filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
            else:
                st.warning("Date column in dataframe is not in datetime format. Cannot apply date filter.")


        if selected_amount_range and pd.api.types.is_numeric_dtype(invoice_df['total_amount']):
             filtered_df = filtered_df[(filtered_df['total_amount'] >= selected_amount_range[0]) & (filtered_df['total_amount'] <= selected_amount_range[1])]


        st.dataframe(filtered_df, use_container_width=True)

        st.subheader("Basic Visualizations")
        if not filtered_df.empty:
            # Invoices by Vendor (Top N)
            st.write("##### Invoice Count by Vendor (Top 10)")
            vendor_counts = filtered_df['vendor'].value_counts().nlargest(10)
            if not vendor_counts.empty:
                st.bar_chart(vendor_counts)
            else:
                st.write("No data to display for vendor counts.")

            # Total Amount by Vendor (Top N)
            if pd.api.types.is_numeric_dtype(filtered_df['total_amount']):
                st.write("##### Total Invoice Amount by Vendor (Top 10)")
                vendor_totals = filtered_df.groupby('vendor')['total_amount'].sum().nlargest(10)
                if not vendor_totals.empty:
                    st.bar_chart(vendor_totals)
                else:
                    st.write("No data to display for vendor totals.")
            else:
                st.write("Total amount data not suitable for numeric aggregation for this chart.")

        else:
            st.write("No data after filtering to display visualizations.")
    else:
        st.warning("Invoice data is empty. Please check the data source.")


with tab2:
    st.header("Neo4j Graph Visualizer (Simulated)")
    st.write("""
    This section would showcase relationships derived from your data (e.g., vendors, invoices, contract terms).
    For an MVP, we can simulate this with a static image or a simplified interactive graph.
    In a full version, this could connect to a live Neo4j instance.
    """)

    # Simulating graph data
    st.write("#### Example: Vendor -> Invoice -> Product/Service Relationship")
    st.image("https://dist.neo4j.com/wp-content/uploads/20180423164259/visualization-dashboard-data-Full.png",
             caption="Simulated graph visualization (replace with actual graph later)",
             use_column_width=True)
    # You could use libraries like streamlit-agraph for a more interactive feel with dummy data:
    # from streamlit_agraph import agraph, Node, Edge, Config
    # nodes = [Node(id="VendorA", label="Vendor A", size=25, shape="diamond"), Node(id="Invoice123", label="INV-123"), Node(id="ProductX", label="Product X")]
    # edges = [Edge(source="VendorA", target="Invoice123", type="CURVE_SMOOTH"), Edge(source="Invoice123", target="ProductX", type="CURVE_SMOOTH")]
    # config = Config(width=750, height=300, directed=True, physics=True, hierarchical=False)
    # agraph(nodes=nodes, edges=edges, config=config)

    st.write("#### Potential Queries (Simulated Interactively):")
    query_options = ["Show all invoices for Vendor X", "Find commonalities between Vendor Y and Vendor Z", "Trace products back to originating invoices"]
    selected_graph_query = st.selectbox("Select a graph query example:", query_options, key="graph_query")

    if selected_graph_query == "Show all invoices for Vendor X":
        st.write("Simulating: Displaying a subgraph of Vendor X and its connected invoices...")
        # (display a relevant small image or mock data table)
    elif selected_graph_query == "Find commonalities between Vendor Y and Vendor Z":
        st.write("Simulating: Highlighting shared products/services or intermediate entities between Vendor Y and Z...")


with tab3:
    st.header("GNN Compliance Dashboard (Simulated)")
    st.write("""
    This dashboard visualizes compliance predictions from a GNN model.
    It can show overall compliance, drill down into specifics, and simulate querying.
    """)
    
    simulated_compliance_df = pd.DataFrame() # Initialize to prevent NameError if invoice_df is empty

    if not invoice_df.empty:
        sample_size = min(20, len(invoice_df)) 
        if sample_size > 0:
            simulated_compliance_df = invoice_df.sample(n=sample_size).copy()
            simulated_compliance_df['compliance_status'] = np.random.choice(["Compliant", "Non-Compliant", "Review Needed"], size=sample_size)
            simulated_compliance_df['confidence_score'] = np.random.rand(sample_size) * 0.4 + 0.6 
            simulated_compliance_df['reason_for_flag'] = [
                np.random.choice(["Unusual payment terms", "Missing signature", "Exceeds budget", ""])
                if status == "Non-Compliant" or status == "Review Needed" else ""
                for status in simulated_compliance_df['compliance_status']
            ]

            st.subheader("Overall Compliance Status")
            compliance_counts = simulated_compliance_df['compliance_status'].value_counts()
            if not compliance_counts.empty:
                # Using Plotly directly for pie chart as st.plotly_chart expects a Plotly figure
                import plotly.express as px
                fig = px.pie(values=compliance_counts.values, 
                             names=compliance_counts.index, 
                             title='Compliance Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No compliance data to display for pie chart.")

            st.subheader("Detailed Compliance View")
            st.write("Click column headers to sort. (Simulated data)")
            st.dataframe(simulated_compliance_df[['filename', 'vendor', 'total_amount', 'compliance_status', 'confidence_score', 'reason_for_flag']], use_container_width=True)

            st.subheader("Simulated Compliance Queries")
            compliance_query = st.selectbox(
                "Select a compliance query:",
                ["Show all Non-Compliant invoices",
                 "Show invoices needing review from 'Vendor A'", 
                 "List vendors with high confidence non-compliance"],
                key="compliance_query_select"
            )

            if compliance_query == "Show all Non-Compliant invoices":
                st.write("Displaying invoices marked as 'Non-Compliant':")
                st.dataframe(simulated_compliance_df[simulated_compliance_df['compliance_status'] == 'Non-Compliant'], use_container_width=True)
            elif compliance_query == "Show invoices needing review from 'Vendor A'":
                example_vendor = ""
                if not simulated_compliance_df['vendor'].empty:
                     example_vendor = simulated_compliance_df['vendor'].unique()[0]
                
                st.write(f"Displaying invoices marked as 'Review Needed' for vendor '{example_vendor}':")
                st.dataframe(simulated_compliance_df[(simulated_compliance_df['compliance_status'] == 'Review Needed') & (simulated_compliance_df['vendor'] == example_vendor)], use_container_width=True)
            elif compliance_query == "List vendors with high confidence non-compliance":
                st.write("Displaying vendors with 'Non-Compliant' invoices (Confidence > 0.8):")
                high_confidence_non_compliant = simulated_compliance_df[
                    (simulated_compliance_df['compliance_status'] == 'Non-Compliant') &
                    (simulated_compliance_df['confidence_score'] > 0.8)
                ]
                if not high_confidence_non_compliant.empty:
                    st.dataframe(high_confidence_non_compliant[['vendor', 'filename', 'confidence_score', 'reason_for_flag']].drop_duplicates(subset=['vendor']), use_container_width=True)
                else:
                    st.write("No vendors found with high confidence non-compliance.")
        else:
            st.warning("Not enough data in invoice_df to create a sample for GNN simulation.")
    else:
        st.warning("Invoice data is empty, cannot simulate GNN compliance.")


with tab4:
    st.header("❓ Ad-hoc Query Simulation")
    st.write("""
    This section simulates how users might ask natural language questions about their data
    and receive insights. For the MVP, we'll provide canned responses to pre-defined questions
    or keywords.
    """)

    query_input = st.text_input("Ask a question about your invoices or compliance (e.g., 'Which vendors have the most non-compliant invoices?'):")

    if query_input:
        query_lower = query_input.lower()
        if "non-compliant" in query_lower and "vendor" in query_lower:
            st.info("Simulating response:")
            st.write("Based on the GNN predictions, 'Vendor X' and 'Vendor Y' currently have the highest number of non-compliant invoices. Would you like to see a detailed breakdown?")
            if not simulated_compliance_df.empty and 'compliance_status' in simulated_compliance_df.columns: 
                 non_compliant_vendors = simulated_compliance_df[simulated_compliance_df['compliance_status'] == 'Non-Compliant']['vendor'].value_counts().nlargest(5)
                 if not non_compliant_vendors.empty:
                     st.bar_chart(non_compliant_vendors)
                 else:
                     st.write("No non-compliant data for vendors.")
            else:
                st.write("Simulated GNN data not available for this query. Ensure invoice data is loaded and GNN simulation in Tab 3 has run.")

        elif "total amount" in query_lower and "last quarter" in query_lower:
            st.info("Simulating response:")
            st.write("The total invoiced amount for the last quarter was $X,XXX,XXX. The top contributing vendors were A, B, and C.")
        elif "high risk contracts" in query_lower:
            st.info("Simulating response:")
            st.write("Identifying high-risk contracts based on compliance status and invoice values...")
            st.write("Contracts associated with invoices INV-001 (Vendor Z, $150k, Non-Compliant) and INV-005 (Vendor W, $90k, Review Needed) are flagged as high risk.")
        else:
            st.info("Simulating response:")
            st.write("This is a simulated response. In a full application, your query would be processed to provide relevant data and visualizations.")

    st.markdown("---")
    st.write("Example questions you could try:")
    st.caption("""
    - "What's the average invoice amount this year?"
    - "Show me all invoices from 'PHILIP MORRIS USA' that are non-compliant."
    - "Which vendors have the highest total invoice values?"
    - "Are there any invoices with amounts over $100,000 needing review?"
    """) 