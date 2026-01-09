import pandas as pd

def drop_duplicate_columns(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Initialize an empty list to keep track of columns to drop
    columns_to_drop = []

    # Iterate over each column and compare it with the other columns
    for i in range(len(df.columns)):
        col1 = df.columns[i]
        for j in range(i + 1, len(df.columns)):
            col2 = df.columns[j]
            # If the columns have the same values, mark one of them for dropping
            if df[col1].equals(df[col2]):
                columns_to_drop.append(col2)

    # Drop the identified duplicate columns
    df = df.drop(columns=columns_to_drop)
    
    # Save the resulting DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage:
input_file = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\Total data.csv'    # Replace with your input CSV file path
output_file = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\Final data.csv'  # Replace with your desired output CSV file path

drop_duplicate_columns(input_file, output_file)
