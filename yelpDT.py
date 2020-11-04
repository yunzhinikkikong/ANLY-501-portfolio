#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:23:49 2020

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
############## Preproessing Function
##########################################################


# from https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()




from sklearn.metrics import classification_report as clsr
def evaluate_model(model, X_test_data, y_test_labels):
    y_predicted_labels = model.predict(X_test_data)
    print(clsr(y_test_labels, y_predicted_labels))



####################################
YelpDF_random = pd.read_csv('/Users/nikkkikong/Desktop/ANLY 501/project data/yelpDT.csv')
##########################################################
############## Wordcloud by Label
##########################################################

YelpDF_negative=YelpDF_random.loc[YelpDF_random['tag'] == 'neg']
YelpDF_positive=YelpDF_random.loc[YelpDF_random['tag'] == 'pos']
YelpDF_neutral=YelpDF_random.loc[YelpDF_random['tag'] == 'neu']

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


#########################################################
#############    Decision Trees   #######################
#########################################################


#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
MyDT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)



## perform DT

for i in [1,2]:
    temp1=str("TrainDF"+str(1))
    temp2=str("Train"+str(1)+"Labels")
    temp3=str("TestDF"+str(i))
    temp4=str("Test"+str(i)+"Labels")
    
    ## perform DT
    MyDT.fit(eval(temp1), eval(temp2))
    ## plot the tree
    tree.plot_tree(MyDT)
    plt.show()
    #plt.savefig(temp1)
    feature_names=eval(str(temp1+".columns"))
    dot_data = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=eval(str(temp1+".columns")),  
                      #class_names=MyDT.class_names,  
                      filled=True, rounded=True,  
                      special_characters=True)                                    
    graph = graphviz.Source(dot_data) 
    ## Create dynamic graph name
    tempname=str("TreeGraph" + str(i))
    graph.render(tempname) 
    ## Show the predictions evaluation from the DT on the test set
    DT_pred=MyDT.predict(eval(temp3))
    print(DT_pred)
    print("\nClassification Report:\n")
    evaluate_model(MyDT, eval(temp3), eval(temp4))
    #print("\nActual for DataFrame: ", i, "\n")
    #print(eval(temp2))
    #print("Prediction\n")    
    #print("\nThe confusion matrix is:")
    #print(bn_matrix)
    FeatureImp=MyDT.feature_importances_   
    indices = np.argsort(FeatureImp)[::-1]
    ## print out the important features.....
    for f in range(TrainDF1.shape[1]):
        if FeatureImp[indices[f]] > 0:
            print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
            print ("feature name: ", feature_names[indices[f]])
    top_ten_arg = indices[:10]
    FeaturesT=eval(temp1).columns
    plt.title('Feature Importances')
    plt.barh(range(len(top_ten_arg)), FeatureImp[top_ten_arg], color='b', align='center')
    plt.yticks(range(len(top_ten_arg)), [FeaturesT[i] for i in top_ten_arg])
    plt.xlabel('Relative Importance')
    plt.show()
    ## Show the confusion matrix
    bn_matrix = confusion_matrix(eval(temp4), DT_pred,labels=['neg','neu','pos'])
    plt.figure()
    plot_confusion_matrix(bn_matrix, classes=['neg','neu','pos'],
                      title='Confusion Matrix')
    plt.show()


#########################################################
#########################################################
##  Random Forest for Text Data with CountVectorizer
#########################################################
#########################################################
RF = RandomForestClassifier()
RF.fit(TrainDF1, Train1Labels)
RF_pred=RF.predict(TestDF1)
print("\nClassification Report:\n")
evaluate_model(RF, TestDF1, Test1Labels)
bn_matrix_RF_text = confusion_matrix(Test1Labels, RF_pred,labels=['neg','neu','pos'])
plt.figure()
plot_confusion_matrix(bn_matrix_RF_text, classes=['neg','neu','pos'],
                      title='Confusion matrix for RF')
plt.show()
#print("\nThe confusion matrix is:")
#print(bn_matrix_RF_text)

################# VIS RF---------------------------------
## FEATURE NAMES...................
FeaturesT=TrainDF1.columns
#Targets=StudentTestLabels_Num

figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

tree.plot_tree(RF.estimators_[0],
               feature_names = FeaturesT, 
               #class_names=Targets,
               filled = True)
plt.show()
##save it
#figT.savefig('RF_Tree_Text')  ## creates png

#####------------------> View estimator Trees in RF

figT2, axesT2 = plt.subplots(nrows = 1,ncols = 5,figsize = (10,8), dpi=900)

for index in range(0, 5):
    tree.plot_tree(RF.estimators_[index],
                   feature_names = FeaturesT, 
                   filled = True,
                   ax = axesT2[index])

    axesT2[index].set_title('Estimator: ' + str(index), fontsize = 11)
## Save it
#figT2.savefig('FIVEtrees_RF.png')

#################-------------------------->
## Feature importance in RF
##-----------------------------------------
## Recall that FeaturesT are the columns names - the words in this case.
######
FeatureImpRF=RF.feature_importances_   
indicesRF = np.argsort(FeatureImpRF)[::-1]
## print out the important features.....
for f2 in range(TrainDF1.shape[1]):   ##TrainDF1.shape[1] is number of columns
    if FeatureImpRF[indicesRF[f2]] >= 0.01:
        print("%d. feature %d (%.2f)" % (f2 + 1, indicesRF[f2], FeatureImpRF[indicesRF[f2]]))
        print ("feature name: ", FeaturesT[indicesRF[f2]])
        

## PLOT THE TOP 10 FEATURES...........................
top_ten_arg = indicesRF[:10]
#print(top_ten_arg)
plt.title('Feature Importances Yelp Reviews(CountVectorizer)')
plt.barh(range(len(top_ten_arg)), FeatureImpRF[top_ten_arg], color='b', align='center')
plt.yticks(range(len(top_ten_arg)), [FeaturesT[i] for i in top_ten_arg])
plt.xlabel('Relative Importance')
plt.show()


#########################################################
##
##                 Random Forest for Text Data with TfidfVectorizer
##
#################################################################

RF = RandomForestClassifier()
RF.fit(TrainDF2, Train2Labels)
RF_pred=RF.predict(TestDF2)
print("\nClassification Report:\n")
evaluate_model(RF, TestDF2, Test2Labels)
bn_matrix_RF_text = confusion_matrix(Test2Labels, RF_pred,labels=['neg','neu','pos'])
plt.figure()
plot_confusion_matrix(bn_matrix_RF_text, classes=['neg','neu','pos'],
                      title='Confusion matrix for RF')
plt.show()


################# VIS RF---------------------------------
## FEATURE NAMES...................
FeaturesT=TrainDF2.columns
#Targets=StudentTestLabels_Num

figT, axesT = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

tree.plot_tree(RF.estimators_[0],
               feature_names = FeaturesT, 
               #class_names=Targets,
               filled = True)
##save it
figT.savefig('RF_Tree_Text')  ## creates png

#####------------------> View estimator Trees in RF

figT2, axesT2 = plt.subplots(nrows = 1,ncols = 5,figsize = (10,8), dpi=900)

for index in range(0, 5):
    tree.plot_tree(RF.estimators_[index],
                   feature_names = FeaturesT, 
                   filled = True,
                   ax = axesT2[index])

    axesT2[index].set_title('Estimator: ' + str(index), fontsize = 11)
## Save it
figT2.savefig('FIVEtrees_RF.png')

#################-------------------------->
## Feature importance in RF
##-----------------------------------------
## Recall that FeaturesT are the columns names - the words in this case.
######
FeatureImpRF=RF.feature_importances_   
indicesRF = np.argsort(FeatureImpRF)[::-1]
## print out the important features.....
for f2 in range(TrainDF2.shape[1]):   ##TrainDF1.shape[1] is number of columns
    if FeatureImpRF[indicesRF[f2]] >= 0.01:
        print("%d. feature %d (%.2f)" % (f2 + 1, indicesRF[f2], FeatureImpRF[indicesRF[f2]]))
        print ("feature name: ", FeaturesT[indicesRF[f2]])
        

## PLOT THE TOP 10 FEATURES...........................
top_ten_arg = indicesRF[:10]
#print(top_ten_arg)
plt.title('Feature Importances Yelp Reviews(TfidfVectorizer)')
plt.barh(range(len(top_ten_arg)), FeatureImpRF[top_ten_arg], color='b', align='center')
plt.yticks(range(len(top_ten_arg)), [FeaturesT[i] for i in top_ten_arg])
plt.xlabel('Relative Importance')
plt.show()




