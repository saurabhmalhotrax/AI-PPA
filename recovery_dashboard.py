import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(layout="wide", page_title="Claims Audit Dashboard")

# --- Mock Data Generation ---
def generate_mock_data(num_records=100):
    np.random.seed(42)
    audit_names = [f"AcmeDemo{i:02d}" for i in range(1, 5)]
    transaction_years = [2022, 2023, 2024]
    transaction_months = [f"{i:02d}" for i in range(1, 13)]
    claim_types = ["ICP", "DUPSV", "CRRT", "DUPB", "CM", "OVPI", "ADJQ", "FRTADJ", "COA", "STLCR", "ADJP", "ADVRBT", "DPOA", "ADJR", "DUPDNT"]
    root_causes = ["Client Proc...", "Vendor Pro...", "Vendor Billi...", "Keying Error", "Calculation...", "Multiple Pa..."]
    service_lines = ["GeneralAudit", "StaleCK/CR"]

    data = {
        'Audit Name': np.random.choice(audit_names, num_records),
        'Transaction Year': np.random.choice(transaction_years, num_records),
        'Transaction Month': np.random.choice(transaction_months, num_records),
        'Identification Year': np.random.choice(transaction_years, num_records), # Assuming same as transaction for simplicity
        'Identification Month': np.random.choice(transaction_months, num_records),
        'Claim Type': np.random.choice(claim_types, num_records),
        'Root Cause': np.random.choice(root_causes, num_records),
        'Service Line': np.random.choice(service_lines, num_records, p=[0.9, 0.1]),
        'Claim Amount': np.random.uniform(1000, 500000, num_records),
        'Auditable Spend Rolling 12': np.random.uniform(10**7, 10**10, num_records),
        'Excluded Spend Rolling 12': np.random.uniform(10**5, 10**7, num_records),
        'Auditable Spend in Scope': np.random.uniform(10**7, 10**9, num_records),
        'Excluded from Audit Scope': np.random.uniform(10**5, 10**7, num_records),
        'Audit Completion Percent': np.random.uniform(0, 100, num_records) # For individual records, though gauge is an aggregate
    }
    df = pd.DataFrame(data)
    df['Claim Count'] = 1 # Each record is one claim for this mock data
    return df

mock_df = generate_mock_data()

# --- Sidebar Filters (Placeholders) ---
st.sidebar.header("Filters")
# Use a key for the text_input to store its value in session_state for KPI cards
audit_name_input = st.sidebar.text_input("Audit Name", "AcmeDemo22", key="audit_name_filter")
st.sidebar.selectbox("Transaction Year", ["All"] + sorted(mock_df['Transaction Year'].unique().tolist()))
st.sidebar.selectbox("Transaction Month", ["All"] + sorted(mock_df['Transaction Month'].unique().tolist()))
st.sidebar.selectbox("Identification Year", ["All"] + sorted(mock_df['Identification Year'].unique().tolist()))
st.sidebar.selectbox("Identification Month", ["All"] + sorted(mock_df['Identification Month'].unique().tolist()))
st.sidebar.selectbox("Claim Type", ["All"] + sorted(mock_df['Claim Type'].unique().tolist()))
st.sidebar.selectbox("Root Cause", ["All"] + sorted(mock_df['Root Cause'].unique().tolist()))

# --- Main Dashboard Area ---
st.title("Audit Dashboard")

# --- KPI Row ---
# Calculate KPI values from the mock dataframe (assuming global filters for now)
# For a real app, these would react to the sidebar filters
total_claim_amount = mock_df['Claim Amount'].sum()
claim_count = mock_df['Claim Count'].sum()
auditable_spend_rolling_12 = mock_df['Auditable Spend Rolling 12'].sum() # Taking sum for mock, image implies single value
excluded_spend_rolling_12 = mock_df['Excluded Spend Rolling 12'].sum()
auditable_spend_in_scope = mock_df['Auditable Spend in Scope'].sum()
excluded_from_audit_scope = mock_df['Excluded from Audit Scope'].sum()

# Get the selected Audit Name from sidebar for the sub-label
# In a real app, this would be part of a filtering logic
selected_audit_name_for_kpi = st.session_state.get('audit_name_filter', mock_df['Audit Name'].iloc[0])

kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric(
        label="Claim Amount", 
        value=f"${total_claim_amount:,.0f}", 
        delta=f"{claim_count} Claim Count",
        delta_color="off" # To make delta appear as a sub-label
    )
with kpi_cols[1]:
    st.metric(
        label="Auditable Spend Rolling 12", 
        value=f"${auditable_spend_rolling_12:,.0f}",
        delta=f"1 of 1 ({selected_audit_name_for_kpi})",
        delta_color="off"
    )
with kpi_cols[2]:
    st.metric(
        label="Excluded Spend Rolling 12", 
        value=f"${excluded_spend_rolling_12:,.0f}",
        delta=f"1 of 1 ({selected_audit_name_for_kpi})",
        delta_color="off"
    )
with kpi_cols[3]:
    st.metric(
        label="Auditable Spend in Scope", 
        value=f"${auditable_spend_in_scope:,.0f}",
        delta=f"1 of 1 ({selected_audit_name_for_kpi})",
        delta_color="off"
    )
with kpi_cols[4]:
    st.metric(
        label="Excluded from Audit Scope", 
        value=f"${excluded_from_audit_scope:,.0f}",
        delta=f"1 of 1 ({selected_audit_name_for_kpi})",
        delta_color="off"
    )

st.markdown("---_--") # Visual separator

# --- Charts Row 1 ---
chart_cols1 = st.columns(2)

with chart_cols1[0]:
    st.subheader("Audit Completion Percent")
    # For the gauge, we'll take a global average or a specific value for simplicity in mock data
    # In a real scenario, this might be pre-calculated or filtered based on 'Audit Name'
    avg_completion_percent = mock_df['Audit Completion Percent'].mean()
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_completion_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Completion"}, # Removed % from title, as it's in the gauge
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "green"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps' : [
                 {'range': [0, 50], 'color': 'lightgray'},
                 {'range': [50, 80], 'color': 'lightyellow'}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': avg_completion_percent}
        }))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with chart_cols1[1]:
    st.subheader("Service Line Claim Amount")
    service_line_claims = mock_df.groupby('Service Line')['Claim Amount'].sum().sort_values(ascending=True)
    if not service_line_claims.empty:
        # For horizontal bar chart with st.bar_chart, ensure data is in correct format
        # It expects a DataFrame where index is y-axis and columns are x-axis values.
        st.bar_chart(service_line_claims, height=250) # Let Streamlit handle orientation if possible
    else:
        st.write("No data for Service Line Claim Amount.")

# --- Charts Row 2 ---
chart_cols2 = st.columns(2)

with chart_cols2[0]:
    st.subheader("Root Cause Analysis")
    root_cause_claims = mock_df.groupby('Root Cause')['Claim Amount'].sum().sort_values(ascending=False)
    if not root_cause_claims.empty:
        st.bar_chart(root_cause_claims, height=300)
    else:
        st.write("No data for Root Cause Analysis.")

with chart_cols2[1]:
    st.subheader("Claim Type Analysis")
    claim_type_analysis = mock_df.groupby('Claim Type')['Claim Amount'].sum().sort_values(ascending=False)
    # Displaying top N for Claim Type Analysis as it has many categories
    top_n_claim_types = 10 
    if not claim_type_analysis.empty:
        st.bar_chart(claim_type_analysis.head(top_n_claim_types), height=300)
    else:
        st.write("No data for Claim Type Analysis.")

# --- Claim Amount and Claim Count Totals Table ---
st.subheader("Claim Amount and Claim Count Totals")

# Aggregate data: Group by Audit Name and sum Claim Amount and Claim Count
if not mock_df.empty:
    agg_df = mock_df.groupby('Audit Name').agg(
        {
            'Claim Amount': 'sum',
            'Claim Count': 'sum'
        }
    ).reset_index()

    # Calculate Totals
    total_row = pd.DataFrame({
        'Audit Name': ['Totals'],
        'Claim Amount': [agg_df['Claim Amount'].sum()],
        'Claim Count': [agg_df['Claim Count'].sum()]
    })

    # Combine aggregated data with totals
    display_df = pd.concat([agg_df, total_row], ignore_index=True)
    
    # Format Claim Amount as currency for display
    display_df_styled = display_df.style.format({
        "Claim Amount": "${:,.0f}"
    })
    
    st.dataframe(display_df_styled, use_container_width=True, hide_index=True)
else:
    st.write("No data to display for Claim Amount and Claim Count Totals.")

# Placeholder for content
st.write("Dashboard content will go here.")
st.write("Simulated Data:")
st.dataframe(mock_df.head())

# To run this file: streamlit run recovery_dashboard.py 