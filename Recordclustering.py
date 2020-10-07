#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:46:50 2020

@author: Nikkikong
"""


import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn import preprocessing 
  

 

Foodborne = pd.read_csv('/Users/nikkkikong/Desktop/ANLY 501/project data/Foodborne_forcluster.csv')
Foodborne["Label"]=pd.Categorical(Foodborne["Label"])
ListOfLabel=Foodborne["Label"]

MyDict={}
for i in range(0, len(ListOfLabel)):
       MyDict[i] = ListOfLabel[i]
Foodborne_label=Foodborne.rename(MyDict, axis="index")
# remove Label as column
Foodborne_label=pd.DataFrame(Foodborne_label.iloc[:,0:3])
# Normalize the data set
Foodborne_label=pd.DataFrame(preprocessing.scale(Foodborne_label))


##########################################################
###### KMeans
##########################################################


#####################"Silhouette"############################
from sklearn.metrics import silhouette_score
sil = []
kmax = 6
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(Foodborne_label)
  labels = kmeans.labels_
  sil.append(silhouette_score(Foodborne_label, labels, metric = 'euclidean'))
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
    kmeans.fit(Foodborne_label)
    distortions.append(kmeans.inertia_)

# plot
plt.plot(range(1, 6), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.title("WSS:Elbow")
plt.show()


################# k means with k = 2 #####################
Mymat=Foodborne_label.values
kmeans_object2 = KMeans(n_clusters=2)
kmeans_object2.fit(Mymat)
# Get cluster assignment labels
labels2 = kmeans_object2.labels_
# Format results as a DataFrame
Myresults2 = pd.DataFrame([Foodborne.index,labels2]).T
plt.scatter(Myresults2.iloc[:, 0], Myresults2.iloc[:, 1], s=50, cmap='viridis')
plt.title("K-means with k = 2")
plt.show()
################# k means with k = 3 #####################
kmeans_object3 = KMeans(n_clusters=3)
kmeans_object3.fit(Mymat)
# Get cluster assignment labels
labels3 = kmeans_object3.labels_
# Format results as a DataFrame
Myresults3 = pd.DataFrame([Foodborne.index,labels3]).T
plt.scatter(Myresults3.iloc[:, 0], Myresults3.iloc[:, 1], s=50, cmap='viridis')
plt.title("K-means with k = 3")
plt.show()
################# k means with k = 4 #####################
kmeans_object4 = KMeans(n_clusters=4)
kmeans_object4.fit(Mymat)
# Get cluster assignment labels
labels4 = kmeans_object4.labels_
# Format results as a DataFrame
Myresults4 = pd.DataFrame([Foodborne.index,labels4]).T
plt.scatter(Myresults4.iloc[:, 0], Myresults4.iloc[:, 1], s=50, cmap='viridis')
plt.title("K-means with k = 4")
plt.show()


##########################################################
###### Hclust
##########################################################

X=Foodborne_label
Z = linkage(squareform(np.around(euclidean_distances(X), 3)),"ward")
fig4 = plt.figure(figsize=(8, 5))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4,orientation="top",labels=Foodborne.index)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
plt.title("Euclidean Distance with Ward Linkage")

Z = linkage(squareform(np.around(manhattan_distances(X), 3)),"single")
fig4 = plt.figure(figsize=(8, 5))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4,orientation="top",labels=Foodborne.index)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
plt.title("Manhattan Distance with Single Linkage")


Z = linkage(squareform(np.around(cosine_distances(X), 3)),"complete")
fig4 = plt.figure(figsize=(8, 5))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4,orientation="top",labels=Foodborne.index)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
plt.title("Cosine Distance with Complete Linkage")










