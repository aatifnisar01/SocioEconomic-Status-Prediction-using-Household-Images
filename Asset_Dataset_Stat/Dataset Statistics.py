# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:17:40 2022

@author: Aatif
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Aatif/nerlp_labels_oct2019.csv')


print('=========================== INCOME DESCRIPTION ===========================')
print(df['Annual household income 2018-19 (~june - July)'].describe())

# qcut() creates 5 bin in which no. of elements is roughly same in each bin
# But different sized interval widths

print('=========================== QUINTILES ===========================')
print(pd.qcut(df['Annual household income 2018-19 (~june - July)'], q=5))


print('=========================== QUINTILES GROUP FREQUENCY ===========================')
print(pd.qcut(df['Annual household income 2018-19 (~june - July)'], q=5).value_counts())



# See correlation of income and other assets by KNN or any other means
# Decreasing values in mentioned column
df.corr().sort_values(by=['Annual household income 2018-19 (~june - July)'],ascending=False)


bin_labels_5 = [0,1,2,3,4]

# Create 5 bins using qcut()
# Created a new column "income_class" (multi class with 5 classes) using bins (qcut())
df['income_class'] = pd.qcut(df['Annual household income 2018-19 (~june - July)'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=False)
# Inferred: Having 5 quantiles was the optimal, since the correlation with assets came highest at this seperation
df.head()


df.corr().sort_values(by=['income_class'],ascending=False)

# How many elements in class 0

df[df['income_class']==0]['Annual household income 2018-19 (~june - July)']

# Plots with equal income class division

import matplotlib.pyplot as plt
df[df['income_class']==0]['Annual household income 2018-19 (~june - July)'].hist(bins=50)


df[df['income_class']==1]['Annual household income 2018-19 (~june - July)'].hist(bins=50)


df[df['income_class']==2]['Annual household income 2018-19 (~june - July)'].hist(bins=50)

df[df['income_class']==4]['Annual household income 2018-19 (~june - July)'].hist(bins=50)

