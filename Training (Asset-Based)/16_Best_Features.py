# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:21:06 2023

@author: Aatif
"""

import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('A:/IIT Delhi/AHI/Assets/combined_data.csv')

# Remove Nan
df_without_nan = df.dropna(axis=0)

# Select target variable 
target = df_without_nan['Annual household income 2018-19 (~june - July)']
target = target.abs()

# Select feature variables 
features = df_without_nan.iloc[:, :-1]

# Standardize the features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(features_std, target, test_size=0.2, random_state=42)

# Perform LASSO regression with cross-validation
lasso_cv = LassoCV(cv=5)  # Specify the number of cross-validation folds
lasso_cv.fit(X_train, y_train)

# Get the selected alpha value
selected_alpha = lasso_cv.alpha_

# Perform LASSO regression with the selected alpha value
lasso = Lasso(alpha=selected_alpha)
lasso.fit(X_train, y_train)

# Get the coefficients and sort them in descending order of magnitude
coefficients = pd.Series(lasso.coef_, index=features.columns)
sorted_coefficients = coefficients.abs().sort_values(ascending=False)

# Select top 16 assets based on highest coefficients
selected_assets = sorted_coefficients.head(16).index.tolist()

# Calculate R-squared value on the test set
r_squared = lasso.score(X_test, y_test)
print("R-squared value:", r_squared)



# **************************************** ON 16 Assets *********************

# =============================================================================
# # Select only the columns present in selected_assets
# X = df_without_nan[selected_assets]
# y = target
# 
# # Perform train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Perform LASSO regression with cross-validation
# lasso_cv = LassoCV(cv=5)  # Specify the number of cross-validation folds
# lasso_cv.fit(X_train, y_train)
# 
# # Get the selected alpha value
# selected_alpha = lasso_cv.alpha_
# 
# # Perform LASSO regression with the selected alpha value
# lasso = Lasso(alpha=selected_alpha)
# lasso.fit(X, y)
# 
# 
# # Calculate R-squared value on the test set
# r_squared = lasso.score(X_test, y_test)
# print("R-squared value:", r_squared)
# 
# 
# 
# # LinearRegression
# linear_regression = LinearRegression()
# linear_regression.fit(X_train, y_train)
# r_squared = linear_regression.score(X_test, y_test)
# print("R-squared:", r_squared)
# =============================================================================

