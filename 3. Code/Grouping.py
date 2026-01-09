import pandas as pd
import os

# Step 1: Load the Data
def load_data(file_paths):
    data_frames = [pd.read_csv(file) for file in file_paths]
    return data_frames

# Step 2: Combine the Data
def combine_data(data_frames):
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

# Step 3: Identify Operating Condition 1
def get_operating_condition_column(data_frame, column_name='opset3'):
    if column_name in data_frame.columns:
        return column_name
    else:
        raise ValueError(f"Column {column_name} not found in DataFrame")

# Step 4: Group the Data by Operating Condition 1
def group_data_by_condition(combined_data, condition_column):
    grouped_data = combined_data.groupby(condition_column)
    return grouped_data

# Step 5: Save or Output the Grouped Data
def save_grouped_data(grouped_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for condition, group in grouped_data:
        # Create a filename based on the operating condition
        condition_str = str(condition)
        # Replace any invalid filename characters
        condition_str = "".join([c if c.isalnum() or c in (' ', '.', '_') else '_' for c in condition_str])
        output_file = os.path.join(output_dir, f'group_{condition_str}.csv')
        group.to_csv(output_file, index=False)
        print(f"Saved grouped data to {output_file}")

# Example Usage
file_paths = [
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1 Opset\Test_15\Features_15.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1 Opset\Test_19\Features_19.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1 Opset\Test_28\Features_28.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1 Opset\Test_30\Features_30.csv',
    r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Data\Test Features Set 1 Opset\Test_46\Features_46.csv',
]

data_frames = load_data(file_paths)
combined_data = combine_data(data_frames)
condition_column = get_operating_condition_column(combined_data, 'opset3')
grouped_data = group_data_by_condition(combined_data, condition_column)
save_grouped_data(grouped_data, 'grouped_data_output')

print("Data has been grouped and saved.")
save_grouped_data(grouped_data, r'D:\Mechatronics\study material\Guru notes\Studienarbeit\Grouped')
