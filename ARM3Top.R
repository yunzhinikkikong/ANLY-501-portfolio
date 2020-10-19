library(tidyverse)
library(viridis)
library(arules)
library(TSP)
library(data.table)
library(devtools)
library(purrr)
library(tidyr)
library(arulesViz)

# Load the transaction data
## https://github.com/yunzhinikkikong/ANLY-501-portfolio/blob/master/UpdatedTNegFile.csv
ReviewTrans <- read.transactions("UpdatedTNegFile.csv", sep =",", 
                                 format("basket"),  rm.duplicates = TRUE)
ReviewTrans_rules = arules::apriori(ReviewTrans, 
                                    parameter = list(support=.01, conf=0.01, minlen=2))
inspect(ReviewTrans_rules)

##  SOrt by Conf
SortedRules_conf <- sort(ReviewTrans_rules, by="confidence", decreasing=TRUE)
inspect(SortedRules_conf[1:15])
## Sort by Sup
SortedRules_sup <- sort(ReviewTrans_rules, by="support", decreasing=TRUE)
inspect(SortedRules_sup[1:15])
## Sort by Lift
SortedRules_lift <- sort(ReviewTrans_rules, by="lift", decreasing=TRUE)
inspect(SortedRules_lift[1:15])