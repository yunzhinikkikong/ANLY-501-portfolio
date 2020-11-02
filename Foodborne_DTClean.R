library(tidyverse)
# read the CSV file into R
Foodborne<-read.csv("NationalOutbreakPublicDataTool.csv",header = TRUE)
# dealing with missing value and lots of "0" in initial data set
all_na <- function(x) any(!is.na(x))
Foodborne<-Foodborne%>% select_if(all_na)
# Remove unnecessary column
Foodborne<-Foodborne[,-c(1,2,3,4,6,14,15,16)]
# delete rows with missing value
Foodborne[Foodborne == ""] <- NA
Foodborne<-Foodborne%>% drop_na()
# new level for setting
# Remove punctuation and shorter the length of string in Setting
Foodborne$setting<-Foodborne$Setting%>%str_remove_all("[[:punct:]]")
Foodborne$setting<-Foodborne$setting%>%substr(1,10)
# Store Setting information in the new variable
Foodborne$setting<-ifelse(str_detect(Foodborne$setting,"Restaurant")==TRUE,"restaurant",
                              ifelse(str_detect(Foodborne$setting,"Private")==TRUE,"private residence","other"))

#re-category the `Etiology` variable with these categories:
# Remove punctuation and shorter the length of string in Setting
Foodborne$Etiology<-Foodborne$Etiology%>%substr(1,10)
# Store Setting information in the new variable
Foodborne$Etiology<-ifelse(str_detect(Foodborne$Etiology,"Norovirus")==TRUE,"Norovirus",
                               ifelse(str_detect(Foodborne$Etiology,"Salmonella")==TRUE,"Salmonella",
                                      ifelse(str_detect(Foodborne$Etiology,"Escherichi")==TRUE,"Escherichia","Other")))
Foodborne<-Foodborne[1:300,]
Foodborne$Etiology.Status<-ifelse(str_detect(Foodborne$Etiology.Status,"Confirmed")==TRUE,"Confirmed","Suspected")
Foodborne<-Foodborne[,-c(3,6,8)]                
write.csv(Foodborne,"Foodborne_forDT.csv", row.names = FALSE)