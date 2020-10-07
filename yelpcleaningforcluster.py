#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:36:06 2020

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
from wordcloud import WordCloud
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re   ## for regular expressions
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances
import string

# read the cleaned Yelp reviews data
YelpDF = pd.read_csv('/Users/nikkkikong/Desktop/ANLY 501/project data/yelpdataforclustering.csv')
# here I will factor the "Label" with 3 levels
YelpDF["Label"]=pd.Categorical(YelpDF["Label"])
YelpDF.dtypes
ListOfLabel=YelpDF["Label"]
data = {"ind":[1,2,3,4,5,6,7]*3}
index = pd.DataFrame (data)
Listofnumber=index["ind"]
# index witl labelin
MyDict={}
for i in range(0, len(ListOfLabel)):
       MyDict[i] = ListOfLabel[i]+str(Listofnumber[i])

YelpDF_label=YelpDF.rename(MyDict, axis="index")
# remove Label as column
#YelpDF_label=pd.DataFrame(YelpDF_label.iloc[:,1])

##########################################################
############## Wordcloud by Label
##########################################################

YelpDF_negative=YelpDF_label.loc[YelpDF_label['Label'] == 'neg']
YelpDF_positive=YelpDF_label.loc[YelpDF_label['Label'] == 'pos']
YelpDF_neutral=YelpDF_label.loc[YelpDF_label['Label'] == 'neu']

# Create three strings, remove punctuation and stopwords.
stoplist = stopwords.words('english')
table = str.maketrans(dict.fromkeys(string.punctuation))  

pos_string = []
for t in YelpDF_positive.text:
    pos_string.append(t)
pos_text = pd.Series(pos_string).str.cat(sep=' ')
pos_text=pos_text.lower()
pos_text = pos_text.translate(table)     
clean_pos_list = str([word for word in pos_text.split() if word not in stoplist])

neg_string = []
for t in YelpDF_negative.text:
    neg_string.append(t)
neg_text = pd.Series(neg_string).str.cat(sep=' ')
neg_text=neg_text.lower()
neg_text = neg_text.translate(table)     
clean_neg_list = str([word for word in neg_text.split() if word not in stoplist])



neu_string = []
for t in YelpDF_neutral.text:
    neu_string.append(t)
neu_text = pd.Series(neu_string).str.cat(sep=' ')
neu_text=neu_text.lower()
neu_text = neu_text.translate(table)     
clean_neu_list = str([word for word in neu_text.split() if word not in stoplist])


# plot the WordCloud image

wordcloud_negative = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(clean_neg_list)
plt.figure(figsize = (12, 8), facecolor = None) 
plt.imshow(wordcloud_negative) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

wordcloud_positive = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(clean_pos_list)
plt.figure(figsize = (12, 8), facecolor = None) 
plt.imshow(wordcloud_positive) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

wordcloud_neutral = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(clean_neu_list)
plt.figure(figsize = (12, 8), facecolor = None) 
plt.imshow(wordcloud_neutral) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

##########################################################
###### Vecterize the data set
##########################################################

MyVect=CountVectorizer(input='content',
                        stop_words='english'
                        )
dtm_2=MyVect.fit_transform(YelpDF["text"])
ColumnNames=MyVect.get_feature_names()
DF_Count=pd.DataFrame(dtm_2.toarray(),columns=ColumnNames)
DF_Count=DF_Count.rename(MyDict, axis="index")
Mymat=DF_Count.values

##########################################################
###### KMeans
##########################################################


#####################"Silhouette"############################
from sklearn.metrics import silhouette_score
sil = []
kmax = 6
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(DF_Count)
  labels = kmeans.labels_
  sil.append(silhouette_score(DF_Count, labels, metric = 'euclidean'))
plt.plot(range(2, kmax+1), sil, marker='o')
plt.xlabel('Number of clusters')
plt.title("Silhouette")
plt.show()


#####################"WSS:Elbow"############################
distortions = []
for i in range(1,6):
    kmeans = KMeans(
        n_clusters=i
    )
    kmeans.fit(DF_Count)
    distortions.append(kmeans.inertia_)

# plot
plt.plot(range(1, 6), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.title("WSS:Elbow")
plt.show()




################# k means with k = 2 #####################
kmeans_object2 = KMeans(n_clusters=2)
kmeans_object2.fit(Mymat)
# Get cluster assignment labels
labels2 = kmeans_object2.labels_
# Format results as a DataFrame
Myresults2 = pd.DataFrame([DF_Count.index,labels2]).T
plt.scatter(Myresults2.iloc[:, 0], Myresults2.iloc[:, 1], s=50, cmap='viridis')
plt.title("K-means with k = 2")
plt.show()
################# k means with k = 3 #####################
kmeans_object3 = KMeans(n_clusters=3)
kmeans_object3.fit(Mymat)
# Get cluster assignment labels
labels3 = kmeans_object3.labels_
# Format results as a DataFrame
Myresults3 = pd.DataFrame([DF_Count.index,labels3]).T
plt.scatter(Myresults3.iloc[:, 0], Myresults3.iloc[:, 1], s=50, cmap='viridis')
plt.title("K-means with k = 3")
plt.show()
################# k means with k = 4 #####################
kmeans_object4 = KMeans(n_clusters=4)
kmeans_object4.fit(Mymat)
# Get cluster assignment labels
labels4 = kmeans_object4.labels_
# Format results as a DataFrame
Myresults4 = pd.DataFrame([DF_Count.index,labels4]).T
plt.scatter(Myresults4.iloc[:, 0], Myresults4.iloc[:, 1], s=50, cmap='viridis')
plt.title("K-means with k = 4")
plt.show()

##########################################################
###### Hclust
##########################################################

X=DF_Count
Z = linkage(squareform(np.around(euclidean_distances(X), 3)),"ward")
fig4 = plt.figure(figsize=(8, 5))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4,orientation="top",labels=DF_Count.index)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
plt.title("Euclidean Distance with Ward Linkage")

Z = linkage(squareform(np.around(manhattan_distances(X), 3)),"single")
fig4 = plt.figure(figsize=(8, 5))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4,orientation="top",labels=DF_Count.index)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
plt.title("Manhattan Distance with Single Linkage")


Z = linkage(squareform(np.around(cosine_distances(X), 3)),"complete")
fig4 = plt.figure(figsize=(8, 5))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4,orientation="top",labels=DF_Count.index)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
plt.title("Cosine Distance with Complete Linkage")















