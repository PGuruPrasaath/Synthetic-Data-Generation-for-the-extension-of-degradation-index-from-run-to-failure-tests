import pandas as pd
import os

def group_by_test_id(csv_file_path, output_directory):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get the column name for the 148th column (index 147 since it's zero-indexed)
    test_id_column = df.columns[147]

    # Group by the test ID column
    grouped = df.groupby(test_id_column)
    
    # Save each group to a separate CSV file
    original_file_name = os.path.basename(csv_file_path)
    original_file_name_without_ext = os.path.splitext(original_file_name)[0]
    
    for test_id, group in grouped:
        output_file_name = f"{test_id}_{original_file_name_without_ext}.csv"
        output_file_path = os.path.join(output_directory, output_file_name)
        group.to_csv(output_file_path, index=False)
        print(f"Saved group {test_id} to {output_file_path}")

def process_multiple_files(input_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Process each CSV file in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(input_directory, file_name)
            try:
                group_by_test_id(csv_file_path, output_directory)
            except ValueError as e:
                print(e)

# Usage
input_directory = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Grouping\Normalized features\Final'  # Directory containing multiple CSV files
output_directory = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Grouping\Normalized features\Individual ID test'  # Ensure this directory exists
process_multiple_files(input_directory, output_directory)
