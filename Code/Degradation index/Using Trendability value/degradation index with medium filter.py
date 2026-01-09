import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import median_filter

# Define file paths from the destination folder using raw file names
file_paths = [
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_15\Features_15.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_19\Features_19.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_28\Features_28.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_30\Features_30.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1\Test_46\Features_46.csv',
]

# Function to process each file to calculate the degradation index as a linear model
def process_file(file_path):
    data = pd.read_csv(file_path)
    
    # Locating the parameters in the data variable
    # the Features are in the first 147 columns
    Features = data.iloc[:, 1:148]
    # Time is in the column 150 (out of 151)
    time = data.iloc[:, -2]
    
    # Drop columns where the standard deviation is zero
    Features = Features.loc[:, Features.std() != 0]
    
    # Calculating absolute correlation coefficients of the features and time
    # Pearson Correlation coefficient
    correlation = Features.corrwith(time)
    absolute_correlation = correlation.abs()
    
    # Handling NaN values that might arise during the correlation calculation
    absolute_correlation = absolute_correlation.dropna()
    
    # Converting the absolute correlation series to a DataFrame
    trendability_df = absolute_correlation.reset_index()
    trendability_df.columns = ['Features', 'Trendability_value']
 
    # Sort the features by trendability value in descending order
    trendability_df = trendability_df.sort_values(by='Trendability_value', ascending=False)
    
    # Plotting the features with their trendability
    # Adjust figure size as needed
    plt.figure(figsize=(12, 8))  
    sns.barplot(y='Features', x='Trendability_value', hue='Features', data=trendability_df, palette="viridis", dodge=False, legend=False)
    plt.xlabel('Trendability Value')
    plt.ylabel('Features')
    plt.title('Trendability of Features')
    plt.tight_layout()
    plt.show()
    
    # Selecting features that have a trendability value greater than 0.5
    selected_features = trendability_df.Features[trendability_df.Trendability_value > 0.5]
    selected_features_data = Features[selected_features]
    
    # Standardizing the selected features before further processing
    standardized_features_data = pd.DataFrame(StandardScaler().fit_transform(selected_features_data), columns=selected_features)
    
    # Apply median filter to remove outliers and noise
    filtered_features_data = standardized_features_data.apply(lambda x: median_filter(x, size=3), axis=0)
        
    # Extracting trendability values of the selected features
    features_trendability_values = trendability_df.loc[trendability_df.Features.isin(selected_features), 'Trendability_value'].values.reshape(-1, 1)
    
    # Calculating degradation index as the linear model as dot product of features and time
    degradation_index = filtered_features_data.dot(features_trendability_values).to_numpy()
    
    # Standardize the degradation index
    degradation_index_standardized = StandardScaler().fit_transform(degradation_index)
    normalized_di = MinMaxScaler().fit_transform(degradation_index)
    
    return file_path, time, normalized_di

# Process each file given in the file_path and collect the results
results = []
for file_path in file_paths:
    results.append(process_file(file_path))

# Plotting all normalized degradation indices
plt.figure(figsize=(12, 8))

for file_path, time, normalized_di in results:
    # To have the label as the the respective file name
    file_name = os.path.basename(file_path)

    # Smooth line without markers for viewing the results
    plt.plot(time, normalized_di, linestyle='-', label=file_name)

    # To have a Darker ^ at the end to indicate the failure of the blade
    plt.scatter(time.iloc[-1], normalized_di[-1], color='black', marker='^', s=100, zorder=5)

plt.xlabel('Time')
plt.ylabel('Normalized Degradation Index')
plt.title('Normalized Degradation Index vs Time - with Medium filter with trendability > 0.5')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()