# Source: https://towardsdatascience.com/doing-your-first-sentiment-analysis-in-r-with-sentimentr-167855445132
# Author: Matti Fuchs - https://towardsdatascience.com/@mattifuchs
# Data: http://jmcauley.ucsd.edu/data/amazon/

library(sentimentr)
library(ndjson)
library(sentimentr)
library(ggplot2)

# Load data
current_path <- paste(getwd(), "/data/", sep="" )
df <- stream_in(paste(current_path, "AmazonBeauty.json", sep = ""))
head(df)

# Get sentiment
df$reviewTextProcessed <- get_sentences(df$reviewText)
sentiment <- sentiment_by(df$reviewTextProcessed)

# Get results
summary(sentiment$ave_sentiment)
qplot(sentiment$ave_sentiment,   
      geom="histogram",binwidth=0.1,main="Review Sentiment Histogram")

df$ave_sentiment=sentiment$ave_sentiment
df$sd_sentiment=sentiment$sd
