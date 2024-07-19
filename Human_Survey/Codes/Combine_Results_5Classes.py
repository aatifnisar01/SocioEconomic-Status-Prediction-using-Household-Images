# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:34:44 2023

@author: Aatif
"""


import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guesses_ANnu_3classes_Int.csv")
df2 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Responses/guesses_Interior_3Classes_Nicky.csv")
df3 = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Survey_Old/Int/InteriorCSV.csv")





df1 = df1.drop(columns=['Comments'])
df2 = df2.drop(columns=['Comments'])
df3 = df3.drop(columns=['Comments'])

df1 = df1.drop(columns=['URL'])
df2 = df2.drop(columns=['URL'])
df3 = df3.drop(columns=['Image'])

class_mapping1 = {
     0:'class0',
     1:'class1',
     2:'class2',
     3:'class3',
     4:'class4'
}

replace_dict = {
    0:'low income',
    1:'lower middle income',
    2:'middle income',
    3:'higher middle income',
    4:'high income'
}

df3['Actual'] = df3['Actual'].replace(class_mapping1)
df3['Predicted'] = df3['Predicted'].replace(replace_dict)

class_mapping2= {
    'low income':'class0',
    'lower middle income':'class1',
    'middle income':'class2',
    'higher middle income':'class3',
    'high income':'class4'
}

df1['Predicted'] = df1['Predicted'].replace(class_mapping2)
df2['Predicted'] = df2['Predicted'].replace(class_mapping2)
df3['Predicted'] = df3['Predicted'].replace(class_mapping2)


# Concatenate df1 and df2 along rows
df = pd.concat([df1, df2, df3], ignore_index=True)

df.columns

selected_columns = ['Actual', 'Predicted']
new_df = df[selected_columns]




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






