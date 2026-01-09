import pandas as pd
import numpy as np
import os

# Corrected CSV file path
csv_file_path = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\Total data.csv'

# Load the data from CSV
df = pd.read_csv(csv_file_path)

# Extract features (excluding the first column, which is assumed to be the time index)
features = df.iloc[:, 1:]

# Get the number of samples (rows) and features (columns)
num_samples, num_features = features.shape

# Generate shuffled indices for columns
column_indices = np.arange(num_features)
np.random.shuffle(column_indices)

# Determine the number of columns for each set
train_size = int(0.75 * num_features)
val_size = int(0.20 * num_features)
test_size = num_features - train_size - val_size

# Split the column indices
train_idx = column_indices[:train_size]
val_idx = column_indices[train_size:train_size + val_size]
test_idx = column_indices[train_size + val_size:]

# Split the data based on the shuffled column indices
train_values = features.iloc[:, train_idx]
val_values = features.iloc[:, val_idx]
test_values = features.iloc[:, test_idx]

# Save unaltered validation and testing data as ground truth
train_values.to_csv(r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\training_data.csv', index=False)
val_values.to_csv(r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\validation_data_ground_truth.csv', index=False)
test_values.to_csv(r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\test_data_ground_truth.csv', index=False)

# Function to introduce missing values
def introduce_missing_values(data, missing_fraction):
    data_with_missing = data.copy()
    # Total number of elements in the DataFrame
    total_values = data_with_missing.size
    # Number of elements to be set as NaN
    num_missing = int(total_values * missing_fraction)
    
    # Randomly select indices to set as NaN
    indices = np.random.choice(total_values, num_missing, replace=False)
    
    # Convert flat indices to 2D indices
    rows, cols = np.unravel_index(indices, data_with_missing.shape)
    
    # Set selected indices to NaN
    data_with_missing.values[rows, cols] = np.nan
    
    return data_with_missing

# Introduce missing values (e.g., 10% missing)
missing_fraction = 0.5
validation_data_with_missing = introduce_missing_values(val_values, missing_fraction)
test_data_with_missing = introduce_missing_values(test_values, missing_fraction)

# Save the altered validation and testing data
validation_data_with_missing.to_csv(r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\validation_data_with_missing.csv', index=False)
test_data_with_missing.to_csv(r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\test_data_with_missing.csv', index=False)

# Display a message indicating completion
print("Data has been successfully split by columns, and missing values have been introduced. Files saved to CSV.")
