# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:51:37 2023

@author: Aatif
"""

import pandas as pd

df = pd.read_csv('A:/IIT Delhi/Codes/Dataset Stat/nerlp_labels_oct2019.csv')

# Dictionary to store statistics for each column
statistics = {}

# Iterate over each column
for column_name in df.columns:
    try:
        column_values = pd.to_numeric(df[column_name], errors='coerce')
        mean_value = column_values.mean()
        std_value = column_values.std()
        min_value = column_values.min()
        max_value = column_values.max()

        # Store statistics in the dictionary
        statistics[column_name] = {
            'Mean': mean_value,
            'Standard Deviation': std_value,
            'Minimum Value': min_value,
            'Maximum Value': max_value
        }
    except ValueError:
        print(f"Unable to calculate statistics for column '{column_name}'. Non-numeric values found.")

# Create a DataFrame from the statistics dictionary
statistics_df = pd.DataFrame.from_dict(statistics, orient='index')

# Save the DataFrame to an Excel file
output_file = 'C:/Users/Aatif/OneDrive/Desktop/statistics1.xlsx'

statistics_df.to_excel(output_file)

print(f"Statistics saved to {output_file}.")

