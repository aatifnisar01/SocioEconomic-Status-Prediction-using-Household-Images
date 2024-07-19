# -*- coding: utf-8 -*-
"""
Created on Sun May 28 12:50:49 2023

@author: Aatif
"""

import pandas as pd
import os

df = pd.read_csv("C:/Users/Aatif/OneDrive/Desktop/some_interior_missing.csv")
list_images = os.listdir('C:/AHI Data/Interior_Split8')

# Extract the numbers from the image names in list_images
image_numbers = [int(image.split("_")[0]) for image in list_images]

# Filter the DataFrame based on the condition
df_filtered = df[df["Unique HHD identifier"].isin(image_numbers)]

df_filtered.to_csv("C:/Users/Aatif/OneDrive/Desktop/filtered_data.csv", index=False)

