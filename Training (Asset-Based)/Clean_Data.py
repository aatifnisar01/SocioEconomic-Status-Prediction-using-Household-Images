# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:53:57 2023

@author: Aatif
"""

import pandas as pd

df = pd.read_csv('A:/IIT Delhi/Codes/Dataset Stat/nerlp_labels_oct2019.csv')

# Assets
features = df.iloc[:, 11:]
features.columns

# *************** CLEANING COLUMNS******************** 

# House_ownership_present
features['House_ownership_present'] = features['House_ownership_present'].replace('Owns a house', 1)
features['House_ownership_present'] = features['House_ownership_present'].replace('Does not own house', 0)

# Bank_account_present
features['Bank_account_present'] = features['Bank_account_present'].replace('have bank account', 1)
features['Bank_account_present'] = features['Bank_account_present'].replace("Doesn't have bank account", 0)

# flooring_materials_present
mapping = {
    'natural floor': 1,
    'finished floor': 2,
    'rudimentary floor': 3
}
features['flooring_materials_present'] = features['flooring_materials_present'].replace(mapping)

# roofing_materials_present
mapping = {
    'natural roofing': 1,
    'finished roofing': 2,
    'rudimentary roofing': 3
}
features['roofing_materials_present'] = features['roofing_materials_present'].replace(mapping)

# walls_materials_present
mapping = {
    'natural walls': 1,
    'finished walls': 2,
    'rudimentary walls': 3
}
features['walls_materials_present'] = features['walls_materials_present'].replace(mapping)

# House type_present
mapping = {
    'semi-puuca': 2,
    'pucca': 1,
    'kuchha':3
}
features['House type_present'] = features['House type_present'].replace(mapping)

# Number of_members per room present
features = features.drop("Number of_members per room present", axis=1)

# Separate room for cooking_present
mapping = {
    'No separate cooking room': 0,
    'have separate cooking room': 1,
}
features['Seperate room for cooking_present'] = features['Seperate room for cooking_present'].replace(mapping)

# type of toilet_present
mapping = {
    'flush piped sewer/ septic tank type sanitation facility': 1,
    'pit/dry type sanitation facility': 2,
    'Shared/community toilet': 3,
    'no sanitation facility': 4
}
features['type of toilet_present'] = features['type of toilet_present'].replace(mapping)

# Main source of lightening_present
mapping = {
    'electricity': 1,
    'coal, charcoal or kerosene': 2,
    'liquid petroleum gas or biogas': 3,
    'No source': 4
}
features['Main source of lightening_present'] = features['Main source of lightening_present'].replace(mapping)

# Main fuel for cooking_present
mapping = {
    'electricity, liquid petroleum gas or biogas': 1,
    'coal, charcoal or kerosene': 2,
    'other fuel': 3
}
features['Main fuel for cooking_present'] = features['Main fuel for cooking_present'].replace(mapping)

# source of drinking water_present
mapping = {
    'public tap, hand pump or well': 1,
    'pipe, hand pump, well in residence/ yard/ plot': 2,
    'other water source': 3
}
features['source of drinking water_present'] = features['source of drinking water_present'].replace(mapping)

# House ownership
mapping = {
    'Own house': 1,
}
features['House ownership'] = features['House ownership'].replace(mapping)


# ****************** TARGET VARIABLE **********************
 
target = df['Annual household income 2018-19 (~june - July)']


# ********* Combine Features and Target to form a dataframe and save the csv file

df_combined = pd.concat([features, target], axis=1)

# Save the dataframe as a CSV file
df_combined.to_csv('combined_data.csv', index=False)
