import streamlit as st
import pandas as pd
import numpy as np # For dummy data
import plotly.graph_objects as go # Added for interactive graph
import networkx as nx # Added for graph layout
from streamlit_plotly_events import plotly_events # Added for click events

st.set_page_config(layout="wide")

# Load the invoice data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Invoice specific columns
        if 'total_amount' in df.columns:
            df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Contract specific columns
        if 'max_value' in df.columns:
            df['max_value'] = pd.to_numeric(df['max_value'], errors='coerce')
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

invoice_df = load_data("data/extracted_invoices.csv")
contracts_df = load_data("data/contracts.csv") # Load contracts data

st.title("Invoice Analysis & Compliance Dashboard MVP")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📄 Invoice Data Explorer",
    "🕸️ Neo4j Graph Visualizer (Interactive)", # Updated Tab Name
    "🛡️ GNN Compliance Dashboard (Simulated)",
    "❓ Ad-hoc Query Simulation",
    "🗂️ Duplicates Dashboard",
    "💰 Recovery Dashboard"
])

with tab1:
    st.header("Invoice Data Explorer")
    if not invoice_df.empty:
        st.write("Browse and filter your extracted invoice data.")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            vendors = ["All"] + sorted(invoice_df['vendor'].astype(str).unique().tolist())
            selected_vendor = st.selectbox("Filter by Vendor", vendors, key="tab1_vendor_filter")
        with col2:
            if not invoice_df['date'].isnull().all():
                min_date = invoice_df['date'].dropna().min()
                max_date = invoice_df['date'].dropna().max()
                selected_date_range = st.date_input("Filter by Date Range",
                                                        value=(min_date, max_date),
                                                        min_value=min_date,
                                                        max_value=max_date,
                                                        key="invoice_date_filter")
            else:
                st.warning("Date data is incomplete or missing for range filtering.")
                selected_date_range = None
        with col3:
            if pd.api.types.is_numeric_dtype(invoice_df['total_amount']) and not invoice_df['total_amount'].isnull().all():
                min_amount = invoice_df['total_amount'].dropna().min()
                max_amount = invoice_df['total_amount'].dropna().max()
                selected_amount_range = st.slider("Filter by Total Amount", float(min_amount), float(max_amount), (float(min_amount), float(max_amount)), key="tab1_amount_slider")
            else:
                selected_amount_range = None 
                st.warning("Total amount column is not numeric or is empty and cannot be used for range slider.")

        filtered_df = invoice_df.copy()
        if selected_vendor != "All":
            filtered_df = filtered_df[filtered_df['vendor'] == selected_vendor]

        if selected_date_range and len(selected_date_range) == 2 and not pd.isna(selected_date_range[0]) and not pd.isna(selected_date_range[1]):
            start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
            if pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
                filtered_df = filtered_df[(filtered_df['date'].notna()) & (filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
            else:
                st.warning("Date column in dataframe is not in datetime format. Cannot apply date filter.")

        if selected_amount_range and pd.api.types.is_numeric_dtype(invoice_df['total_amount']):
             filtered_df = filtered_df[(filtered_df['total_amount'].notna()) & (filtered_df['total_amount'] >= selected_amount_range[0]) & (filtered_df['total_amount'] <= selected_amount_range[1])]

        st.dataframe(filtered_df, use_container_width=True)

        st.subheader("Basic Visualizations")
        if not filtered_df.empty:
            st.write("##### Invoice Count by Vendor (Top 10)")
            vendor_counts = filtered_df['vendor'].value_counts().nlargest(10)
            if not vendor_counts.empty:
                st.bar_chart(vendor_counts)
            else:
                st.write("No data to display for vendor counts.")

            if pd.api.types.is_numeric_dtype(filtered_df['total_amount']) and filtered_df['total_amount'].notna().any():
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
    st.header("Interactive Graph Visualizer (Simulated with Real Data Snippet)")
    st.write("""
    This section showcases relationships using a snippet of your actual invoice and contract data.
    Nodes represent Vendors, Invoices, and Contracts.
    Edges show associations: Vendor -> Invoice and Vendor -> Contract.
    (Requires `networkx` library: `pip install networkx`)
    """)

    # --- Dynamic Filters --- 
    st.sidebar.header("Graph Filters")
    show_invoices = st.sidebar.checkbox("Show Invoices", True, key="show_invoices_filter")
    show_contracts = st.sidebar.checkbox("Show Contracts", True, key="show_contracts_filter")
    # Simulate confidence score for filtering if not present
    if not invoice_df.empty and 'confidence_score' not in invoice_df.columns:
        invoice_df['confidence_score'] = np.random.rand(len(invoice_df)) * 0.5 + 0.5 # Random confidence between 0.5 and 1.0

    # Confidence slider - only if invoices are shown and data is available
    min_confidence = 0.0
    if show_invoices and not invoice_df.empty and 'confidence_score' in invoice_df.columns:
        min_confidence = st.sidebar.slider("Minimum Invoice Confidence Score", 0.0, 1.0, 0.0, 0.05, key="min_confidence_filter")
    # --- End Dynamic Filters ---

    if not invoice_df.empty and not contracts_df.empty:
        # --- Data Preparation ---
        N_VENDORS = st.slider("Number of Top Vendors to Display", 1, 10, 3, key="n_vendors_graph")
        top_vendors_invoice = invoice_df['vendor'].astype(str).value_counts().nlargest(N_VENDORS).index.tolist()

        vendors_key = tuple(top_vendors_invoice)
        if 'graph_vendors' not in st.session_state or st.session_state['graph_vendors'] != vendors_key:
            st.session_state['graph_vendors'] = vendors_key
            # Build and cache simulated invoice/contract data once per vendor selection
            sim_inv = invoice_df[invoice_df['vendor'].isin(top_vendors_invoice)].copy()
            sim_ctr = contracts_df[contracts_df['vendor'].isin(top_vendors_invoice)].copy()
            status_choices = ["Compliant", "Non-Compliant", "Review Needed"]
            if not sim_inv.empty:
                sim_inv['compliance_status'] = np.random.choice(status_choices, size=len(sim_inv))
                if 'confidence_score' in invoice_df.columns:
                    sim_inv = sim_inv.merge(invoice_df[['filename','confidence_score']], on='filename', how='left')
                else:
                    sim_inv['confidence_score'] = np.random.rand(len(sim_inv)) * 0.5 + 0.5
            st.session_state['sim_inv'] = sim_inv
            st.session_state['sim_ctr'] = sim_ctr

        sample_invoices = st.session_state['sim_inv']
        sample_contracts = st.session_state['sim_ctr']

        # Ensure 'vendor' columns are strings for consistent node IDs
        sample_invoices['vendor'] = sample_invoices['vendor'].astype(str)
        sample_contracts['vendor'] = sample_contracts['vendor'].astype(str)
        top_vendors_invoice = [str(v) for v in top_vendors_invoice]

        G = nx.Graph()

        # Add Vendor nodes
        for vendor_name in top_vendors_invoice:
            G.add_node(vendor_name, type='vendor', label=vendor_name)

        # Add Invoice nodes and edges from Vendor to Invoice
        for _, row in sample_invoices.iterrows():
            vendor = row['vendor']
            invoice_id_val = f"Inv: {row.get('invoice_no', row['filename'])}_{vendor}" # Ensure more unique ID with vendor
            G.add_node(invoice_id_val, type='invoice', label=f"Inv: {row.get('invoice_no', 'N/A')}", 
                       amount=row.get('total_amount', 0), filename=row['filename'], # Ensure amount is numeric for sizing
                       compliance_status=row.get('compliance_status', 'Unknown'), 
                       confidence_score=row.get('confidence_score', 0.0))
            if vendor in G and invoice_id_val in G: G.add_edge(vendor, invoice_id_val, type='HAS_INVOICE')

        # Add Contract nodes and edges from Vendor to Contract
        for _, row in sample_contracts.iterrows():
            vendor = row['vendor']
            contract_id_val = f"Con: {row['contract_id']}_{vendor}" # Ensure more unique ID with vendor
            G.add_node(contract_id_val, type='contract', label=f"Con: {row['contract_id'][:15]}...", 
                       value=row.get('max_value', 'N/A'), full_contract_id=row['contract_id'],
                       confidence_score=1.0) # Assume contracts have full confidence for filtering
            if vendor in G and contract_id_val in G: G.add_edge(vendor, contract_id_val, type='HAS_CONTRACT')

        if not G.nodes():
            st.warning("No data to display in the graph for the selected vendors or data structure issue.")
            st.stop() # Exit if no nodes

        # --- Filter graph based on selections --- 
        G_filtered = G.copy() # Start with a copy
        nodes_to_remove = []
        for node_id, data in list(G_filtered.nodes(data=True)):
            node_type = data.get('type')
            node_confidence = data.get('confidence_score', 1.0) # Default to 1.0 if not present

            if node_type == 'invoice' and (not show_invoices or node_confidence < min_confidence):
                nodes_to_remove.append(node_id)
            elif node_type == 'contract' and not show_contracts:
                nodes_to_remove.append(node_id)
        
        G_filtered.remove_nodes_from(nodes_to_remove)

        # If all nodes are filtered out, display a message
        if not G_filtered.nodes():
            st.warning("All nodes have been filtered out based on your selections.")
            st.stop() # Exit if no nodes after filtering
        # --- End Filter graph ---

        pos = nx.spring_layout(G_filtered, k=0.5, iterations=50, seed=42) # Use G_filtered, adjusted k to 0.5

        # --- Calculate axis ranges for better fitting ---
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            if x_coords and y_coords: #Ensure not empty
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                padding_x = (max_x - min_x) * 0.10 # 10% padding
                padding_y = (max_y - min_y) * 0.10 # 10% padding
                
                # Handle cases where min/max are the same (e.g., single node)
                if padding_x == 0: padding_x = 0.5 
                if padding_y == 0: padding_y = 0.5

                x_axis_range = [min_x - padding_x, max_x + padding_x]
                y_axis_range = [min_y - padding_y, max_y + padding_y]
            else:
                x_axis_range = None # Default Plotly autorange
                y_axis_range = None
        else:
            x_axis_range = None
            y_axis_range = None
        # --- End Calculate axis ranges ---

        edge_x, edge_y = [], []
        for edge in G_filtered.edges(): # Use G_filtered
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.7, color='#777'), hoverinfo='none', mode='lines')

        node_x, node_y, node_text, node_color, node_size, node_labels = [], [], [], [], [], []
        base_invoice_size = 8
        invoice_size_scaling_factor = 2

        for node_id, node_data in G_filtered.nodes(data=True): # Use G_filtered
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node_data.get('label', str(node_id))[:20]) # Truncated label on node

            text = f"<b>ID:</b> {str(node_id).split('_')[0]}<br><b>Type:</b> {node_data.get('type','N/A')}"
            if node_data.get('type') == 'vendor':
                node_color.append('rgba(60, 120, 180, 0.9)') # Blueish
                node_size.append(20)
            elif node_data.get('type') == 'invoice':
                text += f"<br><b>Amount:</b> {node_data.get('amount', 'N/A')}"
                text += f"<br><b>File:</b> {node_data.get('filename', 'N/A')}"
                text += f"<br><b>Confidence:</b> {node_data.get('confidence_score', 0.0):.2f}"
                
                status = node_data.get('compliance_status', 'Unknown') # Get status directly from node data
                # status = "Unknown" # Original logic before direct status on node
                # matched_invoice_series = sample_invoices[
                #     (sample_invoices['invoice_no'].astype(str) == invoice_filename_from_node_id) |
                #     (sample_invoices['filename'] == invoice_filename_from_node_id)
                # ]
                # if not matched_invoice_series.empty:
                #     status = matched_invoice_series.iloc[0].get('compliance_status', "Unknown")

                color_map = {
                    "Compliant": 'rgba(46, 204, 113, 0.9)',      # Green
                    "Non-Compliant": 'rgba(231, 76, 60, 0.9)',    # Red
                    "Review Needed": 'rgba(241, 196, 15, 0.9)',   # Yellow
                    "Unknown": 'rgba(149, 165, 166, 0.8)'      # Grey
                }
                node_color.append(color_map.get(status, 'rgba(149, 165, 166, 0.8)'))
                # Dynamic node sizing for invoices
                amount = node_data.get('amount', 0) or 0 # ensure it's not None
                node_size.append(base_invoice_size + np.log1p(float(amount)) * invoice_size_scaling_factor if amount > 0 else base_invoice_size)
            elif node_data.get('type') == 'contract':
                text += f"<br><b>Max Value:</b> {node_data.get('value', 'N/A')}"
                text += f"<br><b>Full ID:</b> {node_data.get('full_contract_id', 'N/A')}"
                node_color.append('rgba(180, 60, 60, 0.9)') # Reddish
                node_size.append(15)
            else:
                node_color.append('rgba(150, 150, 150, 0.8)') # Grey
                node_size.append(10)
            node_text.append(text)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            textfont=dict(size=9, color='black'),
            text=node_labels, textposition="bottom center",
            marker=dict(color=node_color, size=node_size, line_width=1.5, line_color='black'),
            customdata=[{'id': nid, 'type': nd.get('type'), 
                         'compliance_status': nd.get('compliance_status', 'Unknown'), 
                         'confidence_score': nd.get('confidence_score', 0.0)}
                         for nid, nd in G_filtered.nodes(data=True)], # Use G_filtered and add more custom data
            hoverinfo='text', hovertext=node_text
        )

        fig = go.Figure(data=[edge_trace, node_trace], # removed legend_traces
                        layout=go.Layout(
                            autosize=True,
                            height=750,
                            showlegend=False, # Hide internal Plotly legend
                            hovermode='closest',
                            margin=dict(b=10,l=5,r=5,t=80),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=x_axis_range),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=y_axis_range),
                            paper_bgcolor='rgba(245,245,245,0.95)',
                            plot_bgcolor='rgba(245,245,245,0.95)'
                        ))
        selected_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            key="interactive_graph_events",
            override_height=800
        )

        if selected_points:
            selected_point_data = selected_points[0] # Get the first clicked point
            custom_data = selected_point_data.get('customdata')

            if custom_data:
                node_id_clicked = custom_data.get('id')
                node_type_clicked = custom_data.get('type')
                
                st.subheader("Clicked Node Details:")
                st.write(f"**Type:** {node_type_clicked}")
                st.write(f"**Full ID:** {node_id_clicked}")

                # Extract the core identifier part (e.g., invoice number or contract ID or vendor name)
                core_id_parts = str(node_id_clicked).split('_')
                core_id = core_id_parts[0].replace("Inv: ", "").replace("Con: ", "")


                if node_type_clicked == 'invoice':
                    # Try to find by invoice_no first, then by filename if invoice_no was N/A during node creation
                    matched_invoice = sample_invoices[
                        (sample_invoices['invoice_no'].astype(str) == core_id) |
                        (sample_invoices['filename'] == core_id)
                    ]
                    if not matched_invoice.empty:
                        invoice_details = matched_invoice.iloc[0]
                        st.write(f"**Vendor:** {invoice_details.get('vendor', 'N/A')}")
                        st.write(f"**Invoice No:** {invoice_details.get('invoice_no', 'N/A')}")
                        st.write(f"**Filename:** {invoice_details.get('filename', 'N/A')}")
                        st.write(f"**Total Amount:** {invoice_details.get('total_amount', 'N/A')}")
                        st.write(f"**Date:** {invoice_details.get('date', 'N/A')}")
                        st.write(f"**Compliance Status (Simulated):** {invoice_details.get('compliance_status', 'N/A')}")
                        st.write(f"**Confidence Score (Simulated):** {invoice_details.get('confidence_score', 'N/A'):.2f}")
                    else:
                        st.warning(f"Could not find details for invoice ID: {core_id} in the current sample_invoices after filtering. The node might represent an older state or different data subset.")
                
                elif node_type_clicked == 'contract':
                    # The contract_id_val was f"Con: {row['contract_id']}_{vendor}"
                    # core_id should be the contract_id here
                    matched_contract = sample_contracts[sample_contracts['contract_id'].astype(str) == core_id]
                    if not matched_contract.empty:
                        contract_details = matched_contract.iloc[0]
                        st.write(f"**Vendor:** {contract_details.get('vendor', 'N/A')}")
                        st.write(f"**Contract ID:** {contract_details.get('contract_id', 'N/A')}")
                        st.write(f"**Max Value:** {contract_details.get('max_value', 'N/A')}")
                        st.write(f"**Start Date:** {contract_details.get('start_date', 'N/A')}")
                        st.write(f"**End Date:** {contract_details.get('end_date', 'N/A')}")
                    else:
                        st.warning(f"Could not find details for contract ID: {core_id}")

                elif node_type_clicked == 'vendor':
                    # core_id is the vendor name
                    st.write(f"**Vendor Name:** {core_id}")
                    st.write("Displaying invoices for this vendor:")
                    vendor_invoices = sample_invoices[sample_invoices['vendor'] == core_id][['invoice_no', 'filename', 'total_amount', 'date', 'compliance_status']]
                    if not vendor_invoices.empty:
                        st.dataframe(vendor_invoices.head())
                    else:
                        st.write("No invoices found for this vendor in the current sample.")
                    
                    st.write("Displaying contracts for this vendor:")
                    vendor_contracts = sample_contracts[sample_contracts['vendor'] == core_id][['contract_id', 'max_value', 'start_date', 'end_date']]
                    if not vendor_contracts.empty:
                        st.dataframe(vendor_contracts.head())
                    else:
                        st.write("No contracts found for this vendor in the current sample.")
            st.markdown("---") # Separator before any detail display

        # --- Manual Legend Below Graph ---
        st.markdown("#### Node Legend")
        col_v, col_c, col_nc, col_rn, col_ctr = st.columns(5)
        col_v.markdown("🔵 Vendor")
        col_c.markdown("🟢 Compliant")
        col_nc.markdown("🔴 Non-Compliant")
        col_rn.markdown("🟡 Review Needed")
        col_ctr.markdown("🟠 Contract")
        st.markdown("---")
    else:
        if invoice_df.empty:
            st.warning("Invoice data (`extracted_invoices.csv`) is not loaded. Cannot display graph.")
        if contracts_df.empty:
            st.warning("Contract data (`contracts.csv`) is not loaded. Cannot display graph.")

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

# Duplicates and Recovery Dashboards
with tab5:
    st.header("Duplicates Dashboard")
    if not invoice_df.empty:
        # identify duplicates by vendor, amount, and date
        dup_groups = invoice_df.groupby(['vendor', 'total_amount', 'date']).filter(lambda x: len(x) > 1)
        num_groups = invoice_df.groupby(['vendor', 'total_amount', 'date']).size().gt(1).sum()
        st.write(f"Found {len(dup_groups)} duplicate invoices across {num_groups} groups.")
        if not dup_groups.empty:
            st.dataframe(dup_groups.sort_values(['vendor', 'date']), use_container_width=True)
    else:
        st.warning("No invoice data loaded.")

with tab6:
    st.header("Recovery Dashboard")
    if not invoice_df.empty:
        # compute potential recovery amounts from duplicates
        group_stats = invoice_df.groupby(['vendor','total_amount','date']).size().reset_index(name='count')
        recovered = group_stats[group_stats['count']>1].assign(
            recovered_amount=lambda df: (df['count'] - 1) * df['total_amount']
        )
        total_recovered = recovered['recovered_amount'].sum()
        st.metric("Total Potential Recovery", f"${total_recovered:,.2f}")
        if not recovered.empty:
            recovered_by_vendor = recovered.groupby('vendor')['recovered_amount'].sum().sort_values(ascending=False)
            st.bar_chart(recovered_by_vendor)
    else:
        st.warning("No invoice data loaded.") 