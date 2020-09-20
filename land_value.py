#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 21:03:39 2020

@author: Nikkikong
"""

# This file includes python code for cleaning the Land value data set
# You can find the raw data set through below link:
# https://github.com/yunzhinikkikong/ANLY-501-portfolio/blob/master/Agricultural_Land_Values.csv


import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

#  Read the data into Python using pandas as a dataframe
land = pd.read_csv('/Users/nikkkikong/Desktop/ANLY 501/project data/Agricultural Land Values.csv' , sep=',', encoding='latin1')
 
# Print first 5 rows
pprint(land[:5])

#####################
# Clean the data
#####################

# Remove rows with Year that are ealier than 2006.

land.drop(land[land['Year']<2007].index,inplace=True)

# label data

land['State']=pd.Categorical(land['State'])
land['LandCategory']=pd.Categorical(land['LandCategory'])
land['Region']=pd.Categorical(land['Region'])
land['Region or State']=pd.Categorical(land['Region or State'])
land.dtypes

# let's check the category level

land['State']
land['LandCategory']
land['Region']
land['Region or State']

myFileName="land_new.csv"
land.to_csv(myFileName,index=0)

# Boxplot
land.boxplot(column=['Acre Value'], by=['Year'])
plt.show()

# The initial data set is pretty clean.
# What I've done from this data set was just narrowing the year range.
# I will leave the data set like this for now. 
# And I might make some adjustment later when exploring this data set.





