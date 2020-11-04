#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:58:13 2020

@author: Nikkikong
"""


import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re   
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn import tree
import graphviz 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import string






##########################################################
############## Data Cleaning
##########################################################



YelpDF = pd.read_csv('/Users/nikkkikong/randomyelpbyyear.csv')
YelpDF.loc[YelpDF['stars'] <3, 'tag'] = "neg"
YelpDF.loc[YelpDF['stars'] ==3, 'tag'] = "neu"
YelpDF.loc[YelpDF['stars'] >3, 'tag'] = "pos"
YelpDF=YelpDF.iloc[:,[5,2]]

# set seed 
np.random.seed(seed=2)  
 
# check the year level
gbr = YelpDF.groupby('tag')
# Get the variable level index
gbr.groups
 
# select 50 data from tag
typicalNDict = {"neg":50,
                "neu":50,
                "pos":50}
 
# randomly subset the dataset 
def typicalsamling(group,typicalNDict):
    name=group.name
    n=typicalNDict[name]
    return group.sample(n=n)
 
YelpDF_random=YelpDF.groupby(
        'tag',group_keys=False
        ).apply(typicalsamling,typicalNDict)
YelpDF_random=YelpDF_random.sort_index(axis = 0) 
YelpDF_random.to_csv('yelpDT.csv', index=False)





##########################################################
###### Vecterize the data set
##########################################################

ListOfLabel=YelpDF_random["tag"]


MyDict={}
for i in range(0, len(ListOfLabel)):
       MyDict[i] = ListOfLabel[i]


STEMMER=PorterStemmer()
# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words
#MyVect=CountVectorizer(input='content',
                        #stop_words='english',
                        #lowercase = True,
                        #tokenizer=MY_STEMMER
                        #)
MyVect=CountVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )

MyVect_IFIDF_STEM=TfidfVectorizer(input='content',
                        analyzer = 'word',
                        stop_words='english',
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True,
                        )

dtm1=MyVect.fit_transform(YelpDF_random["text"])
ColumnNames1=MyVect.get_feature_names()
DF1_Count=pd.DataFrame(dtm1.toarray(),columns=ColumnNames1)
DF1_Count=DF1_Count.rename(MyDict, axis="index")

dtm2=MyVect_IFIDF_STEM.fit_transform(YelpDF_random["text"])
ColumnNames2=MyVect_IFIDF_STEM.get_feature_names()
DF2_Count=pd.DataFrame(dtm2.toarray(),columns=ColumnNames2)
DF2_Count=DF2_Count.rename(MyDict, axis="index")


##########################################################
# remove any number column

def RemoveNums(SomeDF):
    temp=SomeDF
    MyList=[]
    for col in temp.columns:
        #print(col)
        #Logical1=col.isdigit()  ## is a num
        Logical2=str.isalpha(col) ## this checks for anything
        ## that is not a letter
        if(Logical2==False):# or Logical2==True):
            #print(col)
            MyList.append(str(col))
            #print(MyList)       
    temp.drop(MyList, axis=1, inplace=True)
            #print(temp)
            #return temp
       
    return temp
## Call the function ....
FinalDF_STEM=RemoveNums(DF1_Count)
FinalDF_TFIDF_STEM=RemoveNums(DF2_Count)

##########################################################
## Remove columns that contain "-"  HOW TO....
cols = [c for c in FinalDF_STEM.columns if "-" in c[:] ]
FinalDF_STEM=FinalDF_STEM.drop(cols, axis = 1) 
cols = [c for c in FinalDF_TFIDF_STEM.columns if "-" in c[:] ]
FinalDF_TFIDF_STEM=FinalDF_TFIDF_STEM.drop(cols, axis = 1) 



FinalDF_STEM.to_csv('yelpRT_STEM.csv', index=True)
FinalDF_TFIDF_STEM.to_csv('yelpRT_TFIDF_STEM.csv', index=True)
##########################################################



##########################################################
############## Split the datasets into training and testing set
##########################################################

from sklearn.model_selection import train_test_split
import random as rd
rd.seed(2)
TrainDF1, TestDF1 = train_test_split(FinalDF_STEM, test_size=0.2)
TrainDF2, TestDF2 = train_test_split(FinalDF_TFIDF_STEM, test_size=0.2)



# save the label in train test data
Test1Labels=(TestDF1.index).to_series()
Train1Labels=(TrainDF1.index).to_series()
Test2Labels=(TestDF2.index).to_series()
Train2Labels=(TrainDF2.index).to_series()






