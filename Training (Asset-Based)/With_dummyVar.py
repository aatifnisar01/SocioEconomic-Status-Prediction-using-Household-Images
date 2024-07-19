# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:20:19 2023

@author: Aatif
"""


# =============================================================================
# selected_assets = ['M14A5-Cot / Bed _ At present',
#                    'M14A6-Table _ At present',
#                    'M14A18-Washing Machine? At present',
#                    'M14A16-Refrigerator? At present',
#                    'flooring_materials_present',
#                    'Main fuel for cooking_present',
#                    'Main source of lightening_present',
#                    'M14A20-Computer? At present',
#                    'M14A14-Mobile phone _ At present',
#                    'M14A13-Colour television _ At present',
#                    'M14A21-Motorcycle/scooter? At present',
#                    'M14A22-Car/Jeep _ At present',
#                    'M14A17-Air conditioner / Cooler _ At present',
#                    'M14A3-Pressure cooker _ At present',
#                    'M14A9-Sewing machine _ At present',
#                    'House type_present',
#                    ]
# =============================================================================


import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


df = pd.read_csv('A:/IIT Delhi/Codes/Dataset Stat/nerlp_labels_oct2019.csv')

# Assets
features = df.iloc[:, 11:]



# TARGET VARIABLE 
target = df['Annual household income 2018-19 (~june - July)']

# *********************** DATA CLEANING *************************
 
# House_ownership_present
features['House_ownership_present'] = features['House_ownership_present'].replace('Owns a house', 1)
features['House_ownership_present'] = features['House_ownership_present'].replace('Does not own house', 0)

# Bank_account_present
features['Bank_account_present'] = features['Bank_account_present'].replace('have bank account', 1)
features['Bank_account_present'] = features['Bank_account_present'].replace("Doesn't have bank account", 0)

# flooring_materials_present
features = pd.get_dummies(features, columns=['flooring_materials_present'], drop_first=True)

# roofing_materials_present
features = pd.get_dummies(features, columns=['roofing_materials_present'], drop_first=True)

# walls_materials_present
features = pd.get_dummies(features, columns=['walls_materials_present'], drop_first=True)

# House type_present
features = pd.get_dummies(features, columns=['House type_present'], drop_first=True)

# Number of_members per room present
features = features.drop("Number of_members per room present", axis=1)

# Separate room for cooking_present
mapping = {
    'No separate cooking room': 0,
    'have separate cooking room': 1,
}
features['Seperate room for cooking_present'] = features['Seperate room for cooking_present'].replace(mapping)

# type of toilet_present
features = pd.get_dummies(features, columns=['type of toilet_present'], drop_first=True)

# Main source of lightening_present
features = pd.get_dummies(features, columns=['Main source of lightening_present'], drop_first=True)

# Main fuel for cooking_present
features = pd.get_dummies(features, columns=['Main fuel for cooking_present'], drop_first=True)

# source of drinking water_present
features = pd.get_dummies(features, columns=['source of drinking water_present'], drop_first=True)

# House ownership
mapping = {
    'Own house': 1,
}
features['House ownership'] = features['House ownership'].replace(mapping)






# ********* Combine Features and Target to form a dataframe and save the csv file

df_combined = pd.concat([features, target], axis=1)


# Remove Nan
df_without_nan = df_combined.dropna(axis=0)

# Select target variable 
target = df_without_nan['Annual household income 2018-19 (~june - July)']

# Select feature variables 
features = df_without_nan.iloc[:, :-1]

# Standardize the features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(features_std, target, test_size=0.2, random_state=42)

features.columns



# =============================================================================
# # Perform LASSO regression with cross-validation
# lasso_cv = LassoCV(cv=5)  # Specify the number of cross-validation folds
# lasso_cv.fit(X_train, y_train)
# 
# # Get the selected alpha value
# selected_alpha = lasso_cv.alpha_
# 
# # Perform LASSO regression with the selected alpha value
# lasso = Lasso(alpha=selected_alpha)
# lasso.fit(X_train, y_train)
# 
# # Get the coefficients and sort them in descending order of magnitude
# coefficients = pd.Series(lasso.coef_, index=features.columns)
# sorted_coefficients = coefficients.abs().sort_values(ascending=False)
# 
# # Select top 16 assets based on highest coefficients
# selected_assets = sorted_coefficients.head(16).index.tolist()
# 
# # Calculate R-squared value on the test set
# r_squared = lasso.score(X_test, y_test)
# print("R-squared value:", r_squared)
# =============================================================================


# **************************************** ON 16 Assets *********************

# Select only the columns present in selected_assets
X = df_without_nan[selected_assets]
y = target

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform LASSO regression with cross-validation
lasso_cv = LassoCV(cv=5)  # Specify the number of cross-validation folds
lasso_cv.fit(X_train, y_train)

# Get the selected alpha value
selected_alpha = lasso_cv.alpha_

# Perform LASSO regression with the selected alpha value
lasso = Lasso(alpha=selected_alpha)
lasso.fit(X, y)


# Calculate R-squared value on the test set
r_squared = lasso.score(X_test, y_test)
print("R-squared value:", r_squared)



# LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
r_squared = linear_regression.score(X_test, y_test)
print("R-squared:", r_squared)




# ****************************************** ACCURACY ***************************

# ******************* Work on 16 selected assets **********************

# Select only the columns present in selected_assets
df_with_bestAssets = pd.concat([X, y], axis=1)

#df_with_bestAssets = df_without_nan    # For all features

# Classify Target Values to 5 classes
df_with_bestAssets['Annual household income 2018-19 (~june - July)'] = df_with_bestAssets['Annual household income 2018-19 (~june - July)'].abs()
df_sorted = df_with_bestAssets.sort_values(by='Annual household income 2018-19 (~june - July)')

# Create quantiles and replace them with 0, 1, 2, 3, and 4
df_sorted['Annual household income 2018-19 (~june - July)'] = pd.qcut(df_sorted['Annual household income 2018-19 (~june - July)'], q=5, labels=[0, 1, 2, 3, 4])

# Select target variable 
target = df_sorted['Annual household income 2018-19 (~june - July)']

# Select feature variables 
features = df_sorted.iloc[:, :-1]

# Standardize the features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(features_std, target, test_size=0.2, random_state=42)




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
