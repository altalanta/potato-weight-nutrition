#!/usr/bin/env python3
import glob
import os

import pandas as pd


def convert_all_excel_files():
    """Convert ALL Excel files to CSV"""
    input_dir = "/Users/artemisfolle/Downloads/Potato Raw Dato"
    output_dir = "/Users/artemisfolle/Documents/Coding_Projects/potato_csv_data_complete"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all Excel files
    pattern = f"{input_dir}/**/*.xlsx"
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} Excel files to convert")
    
    converted_count = 0
    error_count = 0
    
    for file_path in files:
        try:
            # Extract participant ID and category
            participant_id = os.path.basename(file_path).split()[0]
            category = os.path.basename(os.path.dirname(file_path))
            
            # Read Excel file
            sheets = pd.read_excel(file_path, sheet_name=None)
            
            for sheet_name, df in sheets.items():
                if not df.empty:
                    # Add metadata
                    df['participant_id'] = participant_id
                    df['category'] = category
                    df['sheet_name'] = sheet_name
                    
                    # Save as CSV
                    csv_filename = f"{participant_id}_{sheet_name.replace(' ', '_')}.csv"
                    csv_path = os.path.join(output_dir, csv_filename)
                    df.to_csv(csv_path, index=False)
            
            converted_count += 1
            if converted_count % 20 == 0:
                print(f"Converted {converted_count} files...")
                
        except Exception as e:
            error_count += 1
            print(f"Error converting {file_path}: {e}")
    
    print("\nConversion complete:")
    print(f"  Successfully converted: {converted_count} files")
    print(f"  Errors: {error_count} files")
    
    return output_dir

if __name__ == "__main__":
    output_dir = convert_all_excel_files()
    print(f"All CSV files saved to: {output_dir}")