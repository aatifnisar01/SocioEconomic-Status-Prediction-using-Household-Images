# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:54:52 2023

@author: Aatif
"""


import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guesses_Interior_3Classes_Nicky.csv")
df2 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guesses_ANnu_3classes_Int.csv")

# Concatenate df1 and df2 along rows
df = pd.concat([df1, df2], ignore_index=True)

df.columns

selected_columns = ['Actual', 'Predicted']
new_df = df[selected_columns]

# Replace values in the specified column
replace_dict = {
    'low income': 'class0',
    'middle income': 'class1',
    'high income': 'class2'
}

new_df['Predicted'] = new_df['Predicted'].replace(replace_dict)


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






