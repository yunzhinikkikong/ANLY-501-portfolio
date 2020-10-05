library(tidyverse)
library(ggplot2)
library(kableExtra)
Yelp<-read.csv("/Users/nikkkikong/randomyelpbyyear.csv",header = TRUE)
Yelp$stars<-as.factor(Yelp$stars)
class(Yelp$stars)
# create a new categorical variable "Label", where star=1 or 2 is negative, star=3 is neutral and star=4 or 5 is positive. 
Yelp$Label<-ifelse(Yelp$stars==1 | Yelp$stars==2,"negative",
                               ifelse(Yelp$stars==3,"neutral","positive"))
Yelp$Label<-as.factor(Yelp$Label)
# Delete columns, keep the text data and "Label" only
Yelp<-Yelp[,-c(1,2,4,5)]
# since the data contain more than 10,000 reviews, I will reduce the data set.
# I will randomly select 21 reviews with fairly percentage of each level in "Label".
set.seed(2)
Yelp_clu<-Yelp%>%group_by(Label)%>%sample_n(7)
Yelp_clu <- Yelp_clu[, c(2,1)]
write.csv(Yelp_clu,"yelpdataforclustering.csv", row.names = FALSE)