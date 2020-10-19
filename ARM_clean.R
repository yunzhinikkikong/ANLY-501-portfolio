library(tidyverse)
library(tokenizers)
# Since the original data is too large, recall that in the Data Cleaning section, 
# I randomly select 1000 rows from each year and created a new .csv file called "randomyelpbyyear.csv" 
# to stored them. In this section I randomly pick 1000 negative reviews from this data set, 
# to be more specific, 500 of them with 1 star and 500 of them with 2 stars, 
# since I defined negative review as star equal to 1 or 2 in the previous section.

Yelp<-read.csv("/Users/nikkkikong/randomyelpbyyear.csv",header = TRUE)
Yelp$stars<-as.factor(Yelp$stars)
# remove rows with reviews are extremely long
Yelp$num_word <- sapply(strsplit(x = Yelp$text, split = ' '), length)
Yelp<-Yelp%>%filter(num_word<100)
yelp_1<-Yelp%>%filter(Yelp$stars == "1") %>% sample_n(., 400) 
yelp_2<-Yelp%>%filter(Yelp$stars == "2") %>% sample_n(., 400) 
Yelp_neg<-rbind(yelp_1,yelp_2)
rm(Yelp)
rm(yelp_1)
rm(yelp_2)
#### Store this data set into csv file for future use
write.csv(Yelp_neg,"/Users/nikkkikong/Desktop/ANLY 501/project data/randomnegyelp.csv", row.names = FALSE)


### transfer the review text into transaction data
##### here I remove stopwords, punctuation, numbers, make word in lowercase. Every word is in a single column.

Yelp_neg<-read.csv("/Users/nikkkikong/Desktop/ANLY 501/project data/randomnegyelp.csv",header = TRUE)
# Delete columns, keep the text data and "Label" only
Yelp_neg<-Yelp_neg[,-c(1,2,4,5,6)]
Yelp_neg<-as.data.frame(Yelp_neg)

## Start the file
TransactionreviewFile = "negreviewResult.csv"

Trans <- file(TransactionreviewFile, open = "a")
for(i in 1:nrow(Yelp_neg)){
  Tokens<-tokenize_words(Yelp_neg$Yelp_neg[i],stopwords = stopwords::stopwords("en"), 
                         lowercase = TRUE,  strip_punct = TRUE, simplify = TRUE)
  cat(unlist(Tokens), "\n", file=Trans, sep=",")
}
close(Trans)
### review the cleaned transaction dataset
NegDF <- read.csv(TransactionreviewFile, 
                  header = FALSE, sep = ",")
head(NegDF)
## Convert all columns to char 
NegDF<-NegDF %>%
  mutate_all(as.character)

# More cleaning, remove digit and remove words with string length shorter than 4 and longer than 9.
MyDF<-NULL
MyDF2<-NULL
for (i in 1:ncol(NegDF)){
  MyList=c() 
  MyList2=c() # each list is a column of logicals ...
  MyList=c(MyList,grepl("[[:digit:]]", NegDF[[i]]))
  MyDF<-cbind(MyDF,MyList)  ## create a logical DF
  MyList2=c(MyList2,(nchar(NegDF[[i]])<4 | nchar(NegDF[[i]])>9))
  MyDF2<-cbind(MyDF2,MyList2) 
  ## TRUE is when a cell has a word that contains digits
}
## For all TRUE, replace with blank
NegDF[MyDF] <- ""
NegDF[MyDF2] <- ""
(head(NegDF,10))

# Now we save the dataframe using the write table command 
write.table(NegDF, file = "UpdatedTNegFile.csv", col.names = FALSE, 
            row.names = FALSE, sep = ",")







