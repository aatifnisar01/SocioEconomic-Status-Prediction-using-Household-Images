# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:05:47 2023

@author: Aatif
"""


import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guesses_Front_Nicky.csv")
df2 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guessesFront_Annu.csv")

# Concatenate df1 and df2 along rows
df12 = pd.concat([df1, df2], ignore_index=True)

df3 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guesses_Int_Annu.csv")
df4 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guesses_Int_Niky.csv")

# Concatenate df1 and df2 along rows
df34 = pd.concat([df3, df4], ignore_index=True)


df = pd.concat([df12, df34], ignore_index=True)

df.columns

selected_columns = ['Actual', 'Predicted']
new_df = df[selected_columns]

# Replace values in the specified column
replace_dict = {
    'low income': 'class0',
    'lower middle income': 'class1',
    'middle income': 'class3',
    'higher middle income': 'class3',
    'high income': 'class4',
    
}

new_df['Predicted'] = new_df['Predicted'].replace(replace_dict)

# Do mapping since we arw combining first 4 classes to representlower income class and the last class to represent higher income class 

# Replace values in the specified column
mapping= {
    'class0': 'class0',
    'class1': 'class0',
    'class2': 'class0',
    'class3': 'class1',
    'class4': 'class1',
}

new_df['Predicted'] = new_df['Predicted'].replace(mapping)
new_df['Actual'] = new_df['Actual'].replace(mapping)


# Calculate accuracy
correct_predictions = (new_df['Actual'] == new_df['Predicted']).sum()
total_predictions = len(new_df)

accuracy = correct_predictions / total_predictions
print("Accuracy:", accuracy)

# Create a confusion matrix
confusion = confusion_matrix(new_df['Actual'], new_df['Predicted'])

# Normalize the confusion matrix
confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

# Set style, color palette, and font scale for the Seaborn plot
sns.set(style='white')
sns.set_palette("pastel")
sns.set(font_scale=1.2)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(confusion_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5, annot_kws={"size": 14})

# Format annotation text with '%' sign
for text in heatmap.texts:
    #text.set_text(text.get_text() + ' %')
    text.set_text(text.get_text())

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Combined', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()






