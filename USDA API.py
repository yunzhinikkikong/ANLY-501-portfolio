#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:24:17 2020

@author: Nikkikong
"""

import requests
import pandas as pd
#import numpy as np

############################
############################
### get the dataset for all commodities's index number
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
### get the commodity data
############################
############################

BaseURL = "https://apps.fas.usda.gov/PSDOnlineDataServices/api/CommodityData/GetCommodityDataByYear?"


##### Since the year range and comodity types that I will use are not sure yet,
## I will run the code to get a test data first. After that, the following list
## will run to fet more data.
#CommodityCode=CommodityData.CommodityCode.tolist()
#year=np.arange(2000, 2019, 1).tolist()

# test range
CommodityCode=["0577400","0011000"]
year=["2010","2001"]
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
      result.to_csv('supplytestData.csv', index=False)
      
    


















