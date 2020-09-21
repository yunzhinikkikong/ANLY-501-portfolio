#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:24:17 2020

@author: Nikkikong
"""

import requests
import pandas as pd
import numpy as np
from pprint import pprint


############################
############################
### Get all commodities's index number
############################
############################

API_URL = "https://apps.fas.usda.gov/PSDOnlineDataServices/api/LookupData/GetCommodities"

headers = {
    "url": "https://apps.fas.usda.gov/PSDOnlineDataServices/api/LookupData/GetCommodities",
    
    "API_KEY": "EA9BB6C4-F05E-4F9D-AA04-B413C781DBC1"
 }
data={"Accept": "application/json"}
response = requests.get(API_URL, json=data,headers=headers)
jsontxt = response.json()

for list in jsontxt:
            CommodityCode = list['CommodityCode']
            CommodityName=list['CommodityName']
            print(CommodityCode," ",CommodityName)
            
CommodityData= pd.DataFrame(data=jsontxt)
CommodityData["CommodityCode"]


############################
############################
### Get the commodity data
############################
############################

BaseURL = "https://apps.fas.usda.gov/PSDOnlineDataServices/api/CommodityData/GetCommodityDataByYear?"


##### I will include all commodity type and year from 2007

CommodityCode=CommodityData.CommodityCode.tolist()
year=np.arange(2007, 2019, 1).tolist()

# test range
#CommodityCode=["0577400","0011000"]
#year=["2010","2001"]
##########
result=pd.DataFrame()
for x in CommodityCode:
  for y in year:
      URLPost = {
           "commodityCode":x,
           "marketYear":y}
      headers = {
         "url" : "https://apps.fas.usda.gov/PSDOnlineDataServices/api/CommodityData/GetCommodityDataByYear?",
         "API_KEY": "EA9BB6C4-F05E-4F9D-AA04-B413C781DBC1"
                }
      data={"Accept": "application/json"}
      response2 = requests.get(BaseURL, URLPost,json=data,headers=headers)
      jsontxt2 = response2.json()
      print(URLPost)
      testData= pd.DataFrame(data=jsontxt2)
      result=result.append(testData)
      PSD=pd.DataFrame(result)
      
############################
############################
### Clean the data set
############################
############################      
    
# Only keep data from the U.S.
PSD_new = PSD[PSD['CountryCode'] == 'US']
# Remove rows with value equal to 0
PSD_new.drop(PSD_new[PSD_new['Value']==0].index,inplace=True)
# print 5 rows of the updated data set
pprint(PSD_new[:5])
# Remove unuseful variable 
PSD_new=PSD_new.drop(columns=['CountryCode', 'CountryName','AttributeId','UnitId'])
# convert below variable into categorical
PSD_new['CommodityCode']=pd.Categorical(PSD_new['CommodityCode'])
PSD_new['CommodityDescription']=pd.Categorical(PSD_new['CommodityDescription'])
PSD_new['AttributeDescription']=pd.Categorical(PSD_new['AttributeDescription'])
PSD_new['UnitDescription']=pd.Categorical(PSD_new['UnitDescription'])
PSD_new.dtypes

# Check the levles of each categprical variable
PSDList=[["Variable:","Level:"],["CommodityCode",PSD_new['CommodityCode'].value_counts().count()],
         ["CommodityDescription",PSD_new['CommodityDescription'].value_counts().count()],
         ['AttributeDescription',PSD_new['AttributeDescription'].value_counts().count()],
         ['UnitDescription',PSD_new['UnitDescription'].value_counts().count()]]

for item in PSDList:   
    print(":",item[0]," "*(20-len(item[0])),":",item[1])




PSD_new.to_csv('PSD_new.csv', index=False)


