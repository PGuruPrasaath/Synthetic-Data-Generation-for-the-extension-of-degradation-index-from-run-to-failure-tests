import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the folder paths
input_folder_path = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Grouping\Normalized features\Collection'
output_folder_path = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Grouping\Normalized features\Final'

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# List all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
    # Load the dataset
    df = pd.read_csv(os.path.join(input_folder_path, csv_file))
    
    # Select columns 1 to 147 for normalization
    columns_to_normalize = df.columns[0:147]
    df_to_normalize = df[columns_to_normalize]
    
    # Fit and transform the selected columns
    normalized_data = scaler.fit_transform(df_to_normalize)
    
    # Create a DataFrame with the normalized data
    normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)
    
    # If there are additional columns that shouldn't be normalized, merge them back
    non_normalized_columns = df.columns[147:]
    if len(non_normalized_columns) > 0:
        remaining_df = df[non_normalized_columns]
        normalized_df = pd.concat([normalized_df, remaining_df], axis=1)
    
    # Save the normalized data to a new CSV file in the output folder
    normalized_csv_file = os.path.join(output_folder_path, csv_file)
    normalized_df.to_csv(normalized_csv_file, index=False)
    
    print(f"Dataset {csv_file} normalization complete and saved to '{normalized_csv_file}'")

print("All datasets normalization complete.")
