# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:16:41 2023

@author: Aatif
"""

import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

df = pd.read_csv('A:/IIT Delhi/AHI/Assets/combined_data.csv')

# Remove Nan
df_without_nan = df.dropna(axis=0)

# ********************** Get best 16 Assets *****************

# =============================================================================
# # Select target variable 
# y = df_without_nan['Annual household income 2018-19 (~june - July)']
# y = y.abs()
# 
# # Select feature variables 
# X = df_without_nan.iloc[:, :-1]
# 
# # Perform LASSO regression with cross-validation
# lasso_cv = LassoCV(cv=5)  # Specify the number of cross-validation folds
# lasso_cv.fit(X, y)
# 
# # Get the selected alpha value
# selected_alpha = lasso_cv.alpha_
# 
# # Perform LASSO regression with the selected alpha value
# lasso = Lasso(alpha=selected_alpha)
# lasso.fit(X, y)
# 
# # Get the coefficients and sort them in descending order of magnitude
# coefficients = pd.Series(lasso.coef_, index=X.columns)
# sorted_coefficients = coefficients.abs().sort_values(ascending=False)
# 
# # Select top 16 assets based on highest coefficients
# selected_assets = sorted_coefficients.head(16).index.tolist()
# 
# # Select only the columns present in selected_assets
# X = df_without_nan[selected_assets]
# 
# df_without_nan = pd.concat([X, y], axis=1)
# =============================================================================


# ******************* Work on 16 selected assets **********************

# Classify Target Values to 5 classes
df_without_nan['Annual household income 2018-19 (~june - July)'] = df_without_nan['Annual household income 2018-19 (~june - July)'].abs()
df_sorted = df_without_nan.sort_values(by='Annual household income 2018-19 (~june - July)')

# Create quantiles and replace them with 0, 1, 2, 3, and 4
df_sorted['Annual household income 2018-19 (~june - July)'] = pd.qcut(df_sorted['Annual household income 2018-19 (~june - July)'], q=2, labels=[0, 1])

# Select target variable 
target = df_sorted['Annual household income 2018-19 (~june - July)']

# Select feature variables 
features = df_sorted.iloc[:, :-1]

# Standardize the features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)





# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_SVM = accuracy_score(y_test, y_pred)


# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_KNN = accuracy_score(y_test, y_pred_knn)


# XGBoost
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_XGB = accuracy_score(y_test, y_pred_xgb)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Create a confusion matrix   (CHOOSE THE PREDICTIONS FROM THE BEST MODEL)
confusion = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix
confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

# Set style, color palette, and font scale for the Seaborn plot
sns.set(style='white')
sns.set_palette("pastel")
sns.set(font_scale=1.2)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(confusion_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5, annot_kws={"size": 14})


plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('SVM', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

