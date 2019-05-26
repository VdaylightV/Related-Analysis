# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:45:25 2019

@author: Administrator
"""

import pandas as pd

import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from apyori import apriori

filename = 'E:/大学课程/AI程序设计/实验部分/实验9 聚类-关联-异常/实验课聚类关联分析/car.data.csv'
data_origin = pd.read_csv(filename, engine = 'python')
data_origin_matrix = data_origin.values

for item in data_origin_matrix:
    if item[0]=='vhigh':
        item[0]='buying_vhigh'
    elif item[0]=='high':
        item[0]='buying_high'
    elif item[0]=='med':
        item[0]='buying_med'
    else:
        item[0]='buying_low'
        
for item in data_origin_matrix:
    if item[1]=='vhigh':
        item[1]='paint_vhigh'
    elif item[1]=='high':
        item[1]='paint_high'
    elif item[1]=='med':
        item[1]='paint_med'
    else:
        item[1]='paint_low'
        
for item in data_origin_matrix:
    if item[2]=='2':
        item[2]='doors_2'
    elif item[2]=='3':
        item[2]='doors_3'
    elif item[2]=='4':
        item[2]='doors_4'
    else:
        item[2]='doors_5more'
        
for item in data_origin_matrix:
    if item[3]=='2':
        item[3]='person_2'
    elif item[3]=='4':
        item[3]='person_4'
    else:
        item[3]='person_more'
    
for item in data_origin_matrix:
    if item[4]=='small':
        item[4]='lug_boot_small'
    elif item[4]=='med':
        item[4]='lug_boot_med'
    else:
        item[4]='lug_boot_big'
        

for item in data_origin_matrix:
    if item[5]=='low':
        item[5]='safety_low'
    elif item[5]=='med':
        item[5]='safety_med'
    else:
        item[5]='safety_high'
        
min_supp = 0.3
min_conf = 0.8

results = list(apriori(transactions = data_origin_matrix, min_support = min_supp,min_confidence=min_conf))

min_supp = 0.2
min_conf = 0.6

results = list(apriori(transactions = data_origin_matrix, min_support = min_supp,min_confidence=min_conf))

from sklearn.cluster import KMeans

filename = 'E:/大学课程/AI程序设计/实验部分/实验9 聚类-关联-异常/实验课聚类关联分析/car.data.csv'
data_origin = pd.read_csv(filename, engine = 'python')
data_mid = data_origin.loc[:,['buying','paint','doors','persons','lug_boot','safety']]
data_mid_matrix=data_mid.values

for item in data_mid_matrix:
    if item[0]=='vhigh':
        item[0]=4
    elif item[0]=='high':
        item[0]=3
    elif item[0]=='med':
        item[0]=2
    else:
        item[0]=1
        
for item in data_mid_matrix:
    if item[1]=='vhigh':
        item[1]=4
    elif item[1]=='high':
        item[1]=3
    elif item[1]=='med':
        item[1]=2
    else:
        item[1]=1
        
for item in data_mid_matrix:
    if item[2]=='2':
        item[2]=2
    elif item[2]=='3':
        item[2]=3
    elif item[2]=='4':
        item[2]=4
    else:
        item[2]=5
        
for item in data_mid_matrix:
    if item[3]=='2':
        item[3]=2
    elif item[3]=='4':
        item[3]=4
    else:
        item[3]=5
    
for item in data_mid_matrix:
    if item[4]=='small':
        item[4]=1
    elif item[4]=='med':
        item[4]=2
    else:
        item[4]=3
        

for item in data_mid_matrix:
    if item[5]=='low':
        item[5]=1
    elif item[5]=='med':
        item[5]=2
    else:
        item[5]=3
        
clf = KMeans(n_clusters=4)
y_pred = clf.fit_predict(data_mid_matrix)

X_person_lst = []
X_safety_lst = []
X_paint_lst = []
X_lugboot_lst = []
Y_lst = []

for item in data_origin_matrix:
    X_person_lst.append(item[3])
    X_safety_lst.append(item[5])
    X_paint_lst.append(item[1])
    X_lugboot_lst.append(item[4])
    
    Y_lst.append(item[6])

ax1 = plt.figure().add_subplot(111)
ax1.scatter(X_person_lst,Y_lst, c=y_pred,marker='o',alpha = 0.2)
ax2 = plt.figure().add_subplot(111)
ax2.scatter(X_safety_lst,Y_lst, c=y_pred,marker='x',alpha = 0.2)
ax3 = plt.figure().add_subplot(111)
ax3.scatter(X_paint_lst,Y_lst, c=y_pred,marker='s',alpha = 0.2)
ax4 = plt.figure().add_subplot(111)
ax4.scatter(X_lugboot_lst,Y_lst, c=y_pred,marker='*',alpha = 0.2)
