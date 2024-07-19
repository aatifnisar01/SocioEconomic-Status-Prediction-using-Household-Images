# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:29:17 2023

@author: Aatif
"""

import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey_New/Survey_Old/Kitchen/Kitchen_CSV.csv")


# Extract the actual labels (ground truth) 
actual_labels = df['Actual']

# Extract the predicted labels 
predicted_labels = df['Predicted']

# Compute the accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)

# Print the accuracy
print("Accuracy:", accuracy)

# Create a confusion matrix
confusion = confusion_matrix(df['Actual'], df['Predicted'])

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
    text.set_text(text.get_text() + ' %')

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
