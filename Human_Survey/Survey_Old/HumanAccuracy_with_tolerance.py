# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:04:25 2023

@author: Aatif
"""


import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey/Int/InteriorCSV.csv")


# Extract the actual labels (ground truth) 
actual_labels = df['Actual']

# Extract the predicted labels 
predicted_labels = df['Predicted']

# Compute the accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)

# Print the accuracy
print("Accuracy:", accuracy)


# Define the tolerance range
tolerance = 1  # Adjust this value as needed

# Compute the accuracy with tolerance
accuracy = sum((abs(actual_labels - predicted_labels) <= tolerance).astype(int)) / len(actual_labels)

# Print the accuracy
print("Accuracy (with tolerance):", accuracy)


# Compute the confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

# Compute the normalized confusion matrix
normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create a DataFrame from the normalized confusion matrix
labels = sorted(set(actual_labels) | set(predicted_labels))
cm_df = pd.DataFrame(normalized_cm, index=labels, columns=labels)

# Create a heatmap of the normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, cmap="Blues")
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()
