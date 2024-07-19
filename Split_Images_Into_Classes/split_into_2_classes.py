# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:58:26 2023

@author: Aatif
"""


import os
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm

list_images = os.listdir('C:/AHI Data/Interior_2Classes')
filename = "C:/Users/Aatif/nerlp_labels_oct2019.csv"
df = pd.read_csv(filename)


df.columns


# Make farm and non farm income all positive

df['Annual net household farm income: 2018-19'] = df['Annual net household farm income: 2018-19'].abs()
df['Annual net household non-farm income: 2018-19'] = df['Annual net household non-farm income: 2018-19'].abs()



# Income = farm + NonFarm
new_col = df['Annual net household farm income: 2018-19'] + df['Annual net household non-farm income: 2018-19']
df = df.drop('Annual household income 2018-19 (~june - July)', axis=1)
df.columns

df.insert(loc=4, column='Annual household income 2018-19 (~june - July)', value=new_col)




# Extract keys and values from df

dict_id_income = {}
for i in range(0,len(df)):
    dict_id_income[str(df.iloc[i, 3])] = df.iloc[i, 4]

#dict_id_income = sorted(dict_id_income.items(), key=lambda x:x[1])


dict_images_income = {}
list_images_incomes = []    # ['key' , 'Income']
list_incomes = []            # Image name e.g. 100021_interior.jgp

for i in list_images:
    # If key in image and .csv are present
    if i.split('_')[0] in dict_id_income.keys():
        
        dict_images_income[i] = dict_id_income[i.split('_')[0]]
        list_images_incomes.append([i.split('_')[0],int(dict_id_income[i.split('_')[0]])])
        list_incomes.append(int(dict_id_income[i.split('_')[0]]))
    else:
        print(i.split('_')[0])


list_incomes.sort()
list_incomes = np.array(list_incomes)




list_images_incomes = sorted(list_images_incomes, key=lambda x:x[1])

list_images_incomes = np.array(list_images_incomes)
#list_images_incomes[1].sort()


# Sort 


#list_incomes = list_images_incomes[:,1]




list_incomes_norm = (list_incomes - list_incomes.mean(axis=0))/(list_incomes.std(axis=0))

# Calculate 20th, 40th, 60th, and 80th percentile, so we can divide data into 5 classes.

y_1 = (np.percentile(list_incomes_norm,50))




train = {0:[],1:[],2:[],3:[],4:[]}
valid = {0:[],1:[],2:[],3:[],4:[]}

li_0 = []
li_1 = []


for i in range(len(list_incomes_norm)):
    if list_incomes_norm[i]<=y_1:
        li_0.append([list_images_incomes[i][0],list_incomes_norm[i]])
    else:
        li_1.append([list_images_incomes[i][0],list_incomes_norm[i]])
        
li = [li_0, li_1]
print ('Sum of lengths: ', len(li_0)+len(li_1), len(list_incomes_norm))

# 60% for train and 20% each for test and validation. 

for i in range(2):
    start = 0
    end = len(li[i])
    train[i] = li[i][0:int(0.7*end)]
    valid[i] = li[i][int(0.7*end):int(end)]

    
print(sum([len(train[i]) for i in range(len(train))]))
print(sum([len(valid[i]) for i in range(len(valid))]))


for i in range(2):
    print('Train folder creation started')
    for x in tqdm(train[i]):
        filename = x[0]
        old_file = 'C:/AHI Data/Interior_2Classes/' + filename + '_HouseInterior_6.jpg'
        new_fold = 'C:/AHI Data/Interior_2Classes/train/class'+str(i)+'/'
        try:
            os.makedirs(new_fold)
            print('Done.')
        except:
            pass
        path = shutil.copy(old_file, new_fold)
        # print (path)
    print('Test folder creation started')

    print('Valid folder creation started')
    for x in tqdm(valid[i]):
        filename = x[0]
        old_file = 'C:/AHI Data/Interior_2Classes/' + filename + '_HouseInterior_6.jpg'
        new_fold = 'C:/AHI Data/Interior_2Classes/valid/class'+str(i)
        try:
            os.makedirs(new_fold)
        except:
            pass
        path = shutil.copy(old_file, new_fold)
    print('Ended copying')