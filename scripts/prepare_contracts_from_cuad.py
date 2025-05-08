import csv
import random
from datetime import datetime, timedelta
import pandas as pd
from datasets import load_dataset

def generate_contract_id(title):
    # Basic anitization and shortening for ID
    return "".join(filter(str.isalnum, title.replace(" ", "_")))[0:50]

def get_vendor_names_from_invoices(invoices_csv_path):
    try:
        invoices_df = pd.read_csv(invoices_csv_path)
        # Assuming vendor column is named 'vendor'
        if 'vendor' in invoices_df.columns:
            return set(invoices_df['vendor'].dropna().unique())
        else:
            print(f"Warning: 'vendor' column not found in {invoices_csv_path}")
            return set()
    except FileNotFoundError:
        print(f"Warning: {invoices_csv_path} not found. Cannot get existing vendor names.")
        return set()
    except Exception as e:
        print(f"Error reading {invoices_csv_path}: {e}")
        return set()

def format_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        # Attempt to parse various common date formats
        # Example: "November 12, 1998", "11/12/1998", "1998-11-12"
        dt_obj = pd.to_datetime(date_str).to_pydatetime()
        return dt_obj.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None # Or handle specific formats if CUAD is consistent

def find_answer(qa_pairs, question_keywords):
    for qa in qa_pairs:
        question = qa['question'].lower()
        if any(keyword.lower() in question for keyword in question_keywords):
            answers = qa['answers']['text']
            if answers:
                return answers[0] # Return the first answer text
    return None

def main():
    output_csv_path = "data/contracts.csv"
    invoices_csv_path = "data/extracted_invoices.csv"
    num_contracts_to_sample = 30

    print("Loading CUAD-QA dataset from Hugging Face...")
    try:
        # Filter for 'Title' and 'Parties' questions directly if possible, or process all
        cuad_qa_dataset = load_dataset("theatticusproject/cuad-qa", split='train')
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load CUAD dataset: {e}")
        return

    existing_vendor_names = get_vendor_names_from_invoices(invoices_csv_path)
    print(f"Found {len(existing_vendor_names)} unique vendor names from invoices.")

    contracts_data = []
    processed_titles = set()
    
    # Group by title to process each contract once
    # The dataset is a stream of Q&A pairs, not pre-grouped by contract.
    # We need to iterate and collect all Q&A for a given title.
    
    # First, let's get unique titles and their corresponding Q&A data
    contracts_qas = {}
    for item in cuad_qa_dataset:
        title = item['title']
        if title not in contracts_qas:
            contracts_qas[title] = {'context': item['context'], 'qas': []}
        contracts_qas[title]['qas'].append({'question': item['question'], 'answers': item['answers']})

    print(f"Found {len(contracts_qas)} unique contract titles in the dataset.")

    # Shuffle and pick a sample of titles
    unique_titles = list(contracts_qas.keys())
    random.shuffle(unique_titles)
    
    sample_titles = unique_titles[:num_contracts_to_sample]
    print(f"Processing {len(sample_titles)} sample contracts...")

    for title in sample_titles:
        if title in processed_titles:
            continue
        
        contract_id = generate_contract_id(title)
        contract_info = contracts_qas[title]
        qas = contract_info['qas']

        # 1. Vendor Name
        vendor_name = None
        parties_answer = find_answer(qas, ["Parties", "Party"])
        if parties_answer:
            # Simplistic: take the first part of the answer, split by common delimiters
            potential_vendors = parties_answer.split(';')[0].split(',')[0].strip()
            if potential_vendors:
                vendor_name = potential_vendors
        
        # Try to use an existing vendor name for linkage
        if existing_vendor_names:
            if vendor_name and vendor_name in existing_vendor_names:
                pass # Use the extracted one if it matches
            else: # Pick a random existing vendor if no match or no extraction
                vendor_name = random.choice(list(existing_vendor_names)) if existing_vendor_names else "Placeholder Vendor Inc."
        elif not vendor_name:
            vendor_name = "Placeholder Vendor Inc."


        # 2. Start Date
        start_date_str = find_answer(qas, ["Agreement Date", "Effective Date"])
        start_date = format_date(start_date_str)
        if not start_date:
            start_date_dt = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365*2)) # Random date in last 2 years
            start_date = start_date_dt.strftime("%Y-%m-%d")
        else:
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")


        # 3. End Date
        end_date_str = find_answer(qas, ["Expiration Date", "Termination Date"])
        end_date = format_date(end_date_str)
        if not end_date:
            # If no end date, make it 1 to 5 years after start_date
            end_date_dt = start_date_dt + timedelta(days=random.randint(365, 365*5))
            end_date = end_date_dt.strftime("%Y-%m-%d")
        
        # Ensure end_date is after start_date
        try:
            if datetime.strptime(end_date, "%Y-%m-%d") <= start_date_dt:
                end_date_dt = start_date_dt + timedelta(days=random.randint(365, 365*2)) # at least 1 year
                end_date = end_date_dt.strftime("%Y-%m-%d")
        except ValueError: # if end_date couldn't be parsed from CUAD and remains placeholder
             end_date_dt = start_date_dt + timedelta(days=random.randint(365, 365*2))
             end_date = end_date_dt.strftime("%Y-%m-%d")


        # 4. Max Value
        max_value = random.randint(10000, 500000)

        contracts_data.append({
            "contract_id": contract_id,
            "vendor_name": vendor_name,
            "max_value": max_value,
            "start_date": start_date,
            "end_date": end_date
        })
        processed_titles.add(title)
        if len(processed_titles) >= num_contracts_to_sample:
            break
    
    # Augment with vendors from invoices not already covered by CUAD processing
    print(f"Augmenting contract data with missing vendors from {invoices_csv_path}...")
    vendors_in_cuad_contracts = set(c['vendor_name'] for c in contracts_data)
    synthetic_contracts_added = 0
    
    if existing_vendor_names:
        invoice_vendors_not_yet_covered = list(existing_vendor_names - vendors_in_cuad_contracts)
        random.shuffle(invoice_vendors_not_yet_covered) # Shuffle to pick random vendors to leave out

        num_to_leave_uncovered = 10
        num_synthetic_to_create = max(0, len(invoice_vendors_not_yet_covered) - num_to_leave_uncovered)
        
        print(f"Total unique invoice vendors: {len(existing_vendor_names)}")
        print(f"Vendors covered by CUAD processing: {len(vendors_in_cuad_contracts.intersection(existing_vendor_names))}")
        print(f"Invoice vendors not covered by CUAD: {len(invoice_vendors_not_yet_covered)}")
        print(f"Aiming to leave {num_to_leave_uncovered} invoice vendors uncovered.")
        print(f"Will create {num_synthetic_to_create} synthetic contracts for other invoice vendors.")

        for i in range(num_synthetic_to_create):
            inv_vendor = invoice_vendors_not_yet_covered[i]
            
            sanitized_vendor_for_id = "".join(filter(str.isalnum, inv_vendor.replace(" ", "_")))[:30]
            contract_id = f"SYNTH_INV_{sanitized_vendor_for_id}_{random.randint(1000,9999)}"
            
            start_date_dt = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 365*2 + 180))
            start_date_str = start_date_dt.strftime("%Y-%m-%d")
            
            end_date_dt = start_date_dt + timedelta(days=random.randint(180, 365*3))
            end_date_str = end_date_dt.strftime("%Y-%m-%d")
            
            max_value = random.randint(5000, 250000)

            contracts_data.append({
                "contract_id": contract_id,
                "vendor_name": inv_vendor,
                "max_value": max_value,
                "start_date": start_date_str,
                "end_date": end_date_str
            })
            synthetic_contracts_added += 1
            # Add to vendors_in_cuad_contracts to avoid processing again if names were similar or duplicated
            # This is more accurately named 'vendors_with_created_contracts' at this stage of the script
            vendors_in_cuad_contracts.add(inv_vendor) 

    print(f"Added {synthetic_contracts_added} synthetic contracts from invoice vendors.")

    if not contracts_data:
        print("No contract data was processed. Exiting.")
        return

    print(f"Writing {len(contracts_data)} contracts to {output_csv_path}...")
    try:
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["contract_id", "vendor_name", "max_value", "start_date", "end_date"])
            writer.writeheader()
            writer.writerows(contracts_data)
        print(f"Successfully wrote contracts to {output_csv_path}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    main() 