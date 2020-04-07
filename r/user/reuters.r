> # Sentiment analysis demo
  
library(tm)
library(SentimentAnalysis)

# Simple example using a sentence. Note use of function > # convertToBinaryResponse() to convert a vector of 
# continuous sentiment scores into a factor object.
sentiment <- analyzeSentiment("My visit to Starbucks today was really lousy.")
convertToBinaryResponse(sentiment)$SentimentQDAP

# More extensive example using the acq data sent from tm
# package, a corpus of 50 Reuters news articles dealing 
# with corporate acquisitions.
data(acq)
sentiment <- analyzeSentiment(acq)  
class(sentiment$NegativityLM)
table(convertToBinaryResponse(sentiment$SentimentLM))

acq[[which.max(sentiment$SentimentLM)]]$meta$heading

summary(sentiment$SentimentLM)

# Visualize density of standardized sentiment variable values
hist(sentiment$SentimentLM, probability=TRUE,
  main="Histogram: Density of Distribution for Standardized Sentiment Variable")
lines(density(sentiment$SentimentLM))

# Calculate the cross-correlation 
cor(sentiment[, c("SentimentLM", "SentimentHE", "SentimentQDAP")])

# Draw a simple line plot to visualize the evolvement of 
# sentiment scores. Helpful when studying a time series 
# of sentiment scores.
plotSentiment(sentiment$SentimentLM, xlab="Reuters News Articles")
