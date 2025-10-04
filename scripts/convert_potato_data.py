#!/usr/bin/env python3
import glob
import os

import pandas as pd


def examine_excel_structure(file_path):
    """Examine the structure of an Excel file"""
    try:
        # Read all sheets
        sheets = pd.read_excel(file_path, sheet_name=None)
        
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Number of sheets: {len(sheets)}")
        
        for sheet_name, df in sheets.items():
            print(f"\nSheet: {sheet_name}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            if not df.empty:
                print("First few rows:")
                print(df.head(3))
                print("\nData types:")
                print(df.dtypes)
        
        return sheets
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def convert_to_csv(input_dir, output_dir):
    """Convert Excel files to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample files from each category
    categories = ['4 Weeks', 'Closed Early', 'Dropped']
    all_data = []
    
    for category in categories:
        pattern = f"{input_dir}/{category}/*.xlsx"
        files = glob.glob(pattern)
        
        if files:
            # Take first few files from each category for analysis
            sample_files = files[:3]  # Limit to 3 files per category for initial analysis
            
            for file_path in sample_files:
                sheets = examine_excel_structure(file_path)
                if sheets:
                    participant_id = os.path.basename(file_path).split()[0]
                    
                    # Save each sheet as CSV
                    for sheet_name, df in sheets.items():
                        if not df.empty:
                            # Add metadata columns
                            df['participant_id'] = participant_id
                            df['category'] = category
                            df['sheet_name'] = sheet_name
                            
                            all_data.append(df)
                            
                            # Save individual CSV
                            csv_filename = f"{participant_id}_{sheet_name.replace(' ', '_')}.csv"
                            csv_path = os.path.join(output_dir, csv_filename)
                            df.to_csv(csv_path, index=False)
                            print(f"Saved: {csv_path}")
    
    # Combine all data if possible
    if all_data:
        try:
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            combined_path = os.path.join(output_dir, "combined_potato_data.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"\nCombined data saved: {combined_path}")
            return combined_path
        except Exception as e:
            print(f"Error combining data: {e}")
            return None
    
    return None

if __name__ == "__main__":
    input_dir = "/Users/artemisfolle/Downloads/Potato Raw Dato"
    output_dir = "/Users/artemisfolle/Documents/Coding_Projects/potato_csv_data"
    
    print("Starting Excel to CSV conversion...")
    combined_file = convert_to_csv(input_dir, output_dir)
    
    if combined_file:
        print(f"\nConversion completed. Combined data available at: {combined_file}")
    else:
        print("\nConversion completed but no combined file created.")