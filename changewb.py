import pandas as pd

# Load the dataset
file_path = 'fully_balanced_labeled_mod2_with_gender.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head())

# Function to check and fix well-being feature
def fix_well_being(row):
    if (18.5 <= row['bmi'] <= 24.9 and
        60 <= row['diastolic'] <= 80 and
        90 <= row['systolic'] <= 120 and
        60 <= row['heartrate_avg'] <= 100 and
        7 <= row['sleep_time'] <= 9 and
        row['steps_count'] >= 10000):
        return 'good'
    else:
        return 'bad'

# Apply the function to check and fix well-being
data['well_being'] = data.apply(fix_well_being, axis=1)

# Display the first few rows of the updated dataframe
print(data.head())

# Save the updated dataframe to a new CSV file
output_file_path = "updated_fully_balanced_labeled_mod2_with_gender.csv"
data.to_csv(output_file_path, index=False)
print(f"Updated dataset saved to {output_file_path}")