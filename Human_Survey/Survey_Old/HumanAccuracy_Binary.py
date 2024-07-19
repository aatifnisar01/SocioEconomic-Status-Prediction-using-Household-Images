# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:04:19 2023

@author: Aatif
"""


import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/Survey/Front/FrontCSV.csv")

# Replace labels in both columns
df['Actual'] = df['Actual'].replace({0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
df['Predicted'] = df['Predicted'].replace({0: 0, 1: 0, 2: 0, 3: 1, 4: 1})


# Extract the actual labels (ground truth) 
actual_labels = df['Actual']

# Extract the predicted labels 
predicted_labels = df['Predicted']

# Compute the accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)

# Print the accuracy
print("Accuracy:", accuracy)

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
