import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import median_filter
import seaborn as sns
import os

# Define file paths
file_paths = [
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_15\Features_15.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_19\Features_19.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_28\Features_28.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_30\Features_30.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_46\Features_46.csv',
]

# Function to process each file
def process_file(file_path):
    data = pd.read_csv(file_path)
    
    # Assuming the last column is the target 'time'
    Features = data.iloc[:, 1:147]
    time = data.iloc[:, -2]
        
    # Calculate initial and final values of the features
    initial_values = Features.iloc[0 , :]
    final_values = Features.iloc[-1 , :]
    
    # Calculate mean and standard deviation of the initial and final values
    diff = (initial_values - final_values) / 2
    diff_abs = diff.abs()
    mean_initial_final = diff_abs.mean()
    print('Size of mean', mean_initial_final.shape)
    std_initial_final = Features.std()
    print('Size of std', std_initial_final.shape)
    
    # Calculate prognosability
    prognosability = np.exp(-std_initial_final / mean_initial_final)
    
    # Convert the prognosability series to a DataFrame
    prognosability_df = prognosability.reset_index()
    prognosability_df.columns = ['Features', 'Prognosability_value']
    
    # Sort the features by prognosability in descending order
    prognosability_df = prognosability_df.sort_values(by='Prognosability_value', ascending=True)
    
    # Plotting the features with their trendability
    # Adjust figure size as needed
    plt.figure(figsize=(12, 8))  
    sns.barplot(y=prognosability_df.Features, x=prognosability_df.Prognosability_value, palette="viridis")
    plt.xlabel('Trendability Value')
    plt.ylabel('Features')
    plt.title('Trendability of Features')
    plt.tight_layout()
    plt.show()

    normalized_features = pd.DataFrame(MinMaxScaler().fit_transform(Features), columns=Features.columns)

    # Selecting features that have a prognosability value greater than 0.6
    selected_features = prognosability_df.Features[prognosability_df.Prognosability_value < 0.9]
    selected_features_data = normalized_features[selected_features]
    

    # Apply median filter to remove outliers
    filtered_features_data = selected_features_data.apply(lambda x: median_filter(x, size=3), axis=0)
    
    # Extracting prognosability values of the selected features
    features_prognosability_values = prognosability_df.loc[prognosability_df.Features.isin(selected_features), 'Prognosability_value'].values.reshape(-1, 1)
    
    # Calculating degradation index
    degradation_index = filtered_features_data.dot(features_prognosability_values)
    normalized_di = MinMaxScaler().fit_transform(degradation_index)
    
    return file_path, time, normalized_di

# Process each file and collect the results
results = []
for file_path in file_paths:
    results.append(process_file(file_path))

# Plotting all normalized degradation indices
plt.figure(figsize=(12, 8))

for file_path, time, normalized_di in results:
    file_name = os.path.basename(file_path)
    plt.plot(time, normalized_di, linestyle='-', label=file_name)  # Smooth line without markers
    plt.scatter(time.iloc[-1], normalized_di[-1], color='black', marker='x', s=100, zorder=5)  # Darker X at the end

plt.xlabel('Time')
plt.ylabel('Normalized Degradation Index')
plt.title('Normalized Degradation Index vs Time for Multiple Files')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
