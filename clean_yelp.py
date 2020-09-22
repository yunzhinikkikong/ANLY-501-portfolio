#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaning the Yelp Review Data Set

@author: Nikkikong
"""

import nltk
import unicodedata
import pandas as pd 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from pprint import pprint
import numpy

# read the csv file obtained last time from json file
YelpDF = pd.read_csv('/Users/nikkkikong/Desktop/yelpreviews.csv')
pprint(YelpDF[:5])
YelpDF = YelpDF.dropna()
YelpDF["year"]=YelpDF["year"].astype(int)
# selcet year range
YelpDF.drop(YelpDF[YelpDF['year']<2007].index,inplace=True)
YelpDF.drop(YelpDF[YelpDF['year']>2019].index,inplace=True)
YelpDF_new=YelpDF
YelpDF_new.to_csv('yelpreviews_new.csv', index=False)


# set seed 
numpy.random.seed(seed=2)  
 
# check the year level
gbr = YelpDF.groupby('year')
# Get the variable level index
gbr.groups
 
# select 100 data from each year
typicalNDict = {2007:1000,
                2008:1000,
                2009:1000, 
                2010:1000, 
                2011:1000, 
                2012:1000, 
                2013:1000, 
                2014:1000, 
                2015:1000, 
                2016:1000, 
                2017:1000, 
                2018:1000, 
                2019:1000}
 
# randomly subset the dataset 
def typicalsamling(group,typicalNDict):
    name=group.name
    n=typicalNDict[name]
    return group.sample(n=n)
 
YelpDF_random=YelpDF.groupby(
        'year',group_keys=False
        ).apply(typicalsamling,typicalNDict)
pprint(YelpDF_random[:2])


review=YelpDF_random.iloc[:,2]
pprint(review[:5])
cleanrandomreview=review
cleanrandomreview.to_csv('cleanrandomreviews.csv', index=False)



nltk.download('punkt')
def normalize(text):
    return  unicodedata.normalize('NFD',text.strip().lower())



def tokenize(text, remove_punct=True):
    tokens = []
    text = normalize(text)
 
    for token in nltk.word_tokenize(text):
        if remove_punct and token in string.punctuation: 
            continue
        else:
            tokens.append(token)
    return tokens

review_words = ''
for text_chunk in review:
    tokens = tokenize(text_chunk)
    review_words += " ".join(tokens)+" "



textfile = open('review_words.txt', 'w')
textfile.write(review_words)
textfile.close()


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(review_words)



# plot the WordCloud image                        
plt.figure(figsize = (12, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

# remove stopword and stem

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stemmer = nltk.stem.SnowballStemmer('english') 

def tokenize(text, remove_punct=False,  remove_stop=False, stem_tokens=False, all_fields=False):
    tokens = []
    text = normalize(text)
 
    for token in nltk.pos_tag(nltk.word_tokenize(text)):
        stem = ''
        token_text = token[0]
        if remove_punct and token_text in string.punctuation: 
            continue
        if remove_stop and token_text.strip().lower() in stopwords:
            continue
        if stem_tokens or all_fields:
            stem = stemmer.stem(token_text)
        if all_fields:
            tokens.append({'token': token_text, 'stem': stem})
        elif stem_tokens:
            tokens.append(stem)
        else:
            tokens.append(token_text)     
    return tokens

review_words_stop = ''
for text_chunk in review:
    tokens = tokenize(text_chunk,remove_punct=True, stem_tokens=True,remove_stop=True)
    review_words_stop += " ".join(tokens)+" "

textfile = open('review_words_stop.txt', 'w')
textfile.write(review_words_stop)
textfile.close()




wordcloud_stop = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(review_words_stop)


# plot the WordCloud image                        
plt.figure(figsize = (12, 8), facecolor = None) 
plt.imshow(wordcloud_stop) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 
      










