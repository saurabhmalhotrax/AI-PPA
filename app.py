import streamlit as st

def main():
    st.set_page_config(page_title="MVP Invoice Auditor", layout="wide")
    st.title("Invoice Auditing MVP - Hello World!")
    st.write("Welcome to the AI-assisted invoice auditing system.")

    # Placeholder for future content
    # st.sidebar.header("Navigation")
    # page = st.sidebar.radio("Go to", ["Upload & Extract", "View Duplicates", "Compliance Check"])

    # if page == "Upload & Extract":
    #     st.header("Upload and Extract Invoice Data")
        # ... upload logic here ...

if __name__ == "__main__":
    main() 