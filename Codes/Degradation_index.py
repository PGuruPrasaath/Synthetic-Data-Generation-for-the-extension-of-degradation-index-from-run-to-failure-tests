import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Define the directory containing the CSV files
directory_path = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Grouping\Normalized features\Individual ID test\Feature_15'

# Output file path to save all degradation indices in one CSV file
output_file_path = r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Training data set\46 _Degradation_Indices.csv'

# Function to process each file to calculate the degradation index as a linear model
def process_file(file_path):
    try:
        data = pd.read_csv(file_path)
        
        # Locating the parameters in the data variable
        # The Features are in the first 147 columns
        Features = data.iloc[:, 1:148]
        # Time is in the column 149
        time = data.iloc[:, 148]
        
        # Fill NaN values with zero to avoid calculation issues
        Features = Features.fillna(0)
        time = time.fillna(0)
        
        # Calculating absolute correlation coefficients of the features and time
        correlation = Features.corrwith(time)
        absolute_correlation = correlation.abs()
        
        # Handle potential NaN values
        absolute_correlation = absolute_correlation.fillna(0)
        
        # Converting the absolute correlation series to a DataFrame
        trendability_df = absolute_correlation.reset_index()
        trendability_df.columns = ['Features', 'Trendability_value']
     
        # Sort the features by trendability value in descending order
        trendability_df = trendability_df.sort_values(by='Trendability_value', ascending=False)
        
        # Selecting features that have a trendability value greater than 0.5
        selected_features = trendability_df.Features[trendability_df.Trendability_value > 0.5]
        selected_features_data = Features[selected_features]
        
        # Apply median filter to remove outliers and noise
        filtered_features_data = selected_features_data.apply(lambda x: median_filter(x, size=3), axis=0)
        
        # Extracting trendability values of the selected features
        features_trendability_values = trendability_df.loc[trendability_df.Features.isin(selected_features), 'Trendability_value'].values.reshape(-1, 1)
        
        # Calculating degradation index as the dot product of features and trendability values
        degradation_index = filtered_features_data.dot(features_trendability_values).values.flatten()
        
        return file_path, time, degradation_index
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Collect all degradation indices into a single DataFrame
all_degradation_indices = pd.DataFrame()

# Process each CSV file in the specified directory and collect the results
results = []
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if file_path.endswith('.csv'):
        result = process_file(file_path)
        if result:
            file_path, time, degradation_index = result
            # Create a DataFrame for this file's results
            result_df = pd.DataFrame({'Time': time, os.path.basename(file_path): degradation_index})
            # Merge with the main DataFrame
            if all_degradation_indices.empty:
                all_degradation_indices = result_df
            else:
                all_degradation_indices = pd.merge(all_degradation_indices, result_df, on='Time', how='outer')
            results.append(result)
        else:
            print(f"Skipping file {file_path} due to processing error.")

# Save all degradation indices to a single CSV file
all_degradation_indices.to_csv(output_file_path, index=False)

# Plotting all degradation indices
plt.figure(figsize=(14, 8))

for file_path, time, degradation_index in results:
    # To have the label as the respective file name
    file_name = os.path.basename(file_path)
    
    # Ensure time and degradation_index are properly aligned
    plt.plot(time, degradation_index, linestyle='-', label=file_name)

    # To have a Darker ^ at the end to indicate the failure of the blade
    plt.scatter(time.iloc[-1], degradation_index[-1], color='black', marker='^', s=100, zorder=5)

plt.xlabel('Time')
plt.ylabel('Degradation Index')
plt.title('Degradation Index vs Time for All Files')
plt.grid(True)
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()

# Show the plot
plt.show()
