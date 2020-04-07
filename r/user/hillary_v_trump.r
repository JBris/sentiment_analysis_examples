# Source: https://medium.com/analytics-vidhya/sentiment-analysis-using-r-c9af723dc57d
# Author: Kafaru simmie - https://medium.com/@kafarusimileoluwa
# Data: https://github.com/simmieyungie/Sentiment-Analysis

library(tidyverse)
library(tidytext)
library(wordcloud)
library(reshape2)

debate <- read.csv("~/data/debate.csv", stringsAsFactors = F)
  
#analyzing the us presidential debate 
#words used largely by candidates
debate %>% 
  group_by(Speaker) %>%
  unnest_tokens(word, Text) %>% #Tokenization 
  group_by(Speaker) %>% 
  anti_join(stop_words) %>% #remove stop words
  count(word, sort = T) %>% 
  mutate(word = str_extract(word, "[a-z]+")) %>% 
  na.omit() %>% 
  filter(n > 30) %>% #Extract words with frequencies > 20
  ggplot(., aes(reorder(word, n), n, fill = Speaker)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  ylab("Score") +
  xlab("Words") + ggtitle("Word Frequency") +
  theme(plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
        axis.title.x = element_text(face = "bold", size = 13),
        axis.title.y = element_text(face = "bold", size = 13))

#Get the sentiments Variation
debate %>% 
  filter(Speaker %in% c("Trump","Clinton")) %>% 
  unnest_tokens(word, Text) %>% 
  anti_join(stop_words) %>% 
  inner_join(get_sentiments("bing")) %>% 
  count(Speaker, index = Line, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>% 
  ggplot(.,aes(index, sentiment, fill = Speaker)) +
  geom_col(show.legend = FALSE, width = 3) +
  facet_wrap(~Speaker, ncol = 18, scales = "free_x") +
  ggtitle("Sentiments Variation") + 
  theme(plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
        axis.title.x = element_text(face = "bold", size = 13),
        axis.title.y = element_text(face = "bold", size = 13))

#plot a comparison of postive and negative words used by participant speakers (Trump vs Clinton)
debate %>% 
  filter(Speaker %in% c("Trump","Clinton")) %>% 
  unnest_tokens(word, Text) %>% 
  anti_join(stop_words) %>% 
  inner_join(get_sentiments("bing")) %>% 
  group_by(sentiment, Speaker) %>% 
  count(word) %>% 
  top_n(10) %>% 
  ggplot(., aes(reorder(word, n), n, fill = Speaker)) +
  geom_col(show.legend = T) +
  coord_flip() +
  facet_wrap(~sentiment, scales = "free_y") +
  xlab("Words") +
  ylab("frequency") +
  ggtitle("Word Usage") +
  theme(plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
        axis.title.x = element_text(face = "bold", size = 13),
        axis.title.y = element_text(face = "bold", size = 13))

debate %>% 
  filter(Speaker %in% c("Trump","Clinton")) %>% 
  unnest_tokens(word, Text) %>% 
  mutate(word = gsub("problems", "problem", word)) %>% 
  inner_join(get_sentiments("bing")) %>% 
  count(word, sentiment) %>% 
  acast(word~sentiment, value.var = "n", fill = 0) %>% 
  comparison.cloud(color = c("#1b2a49", "#00909e"),
                   max.words = 100)
# Note the acast function is from the reshape2 package
# Functions such as comparison.cloud() require you to turn the data frame into a matrix with reshape2’s acast()

#Get speakers with most negative words
debate %>% 
  unnest_tokens(word, Text) %>% 
  anti_join(stop_words) %>% 
  inner_join(get_sentiments("bing")) %>% 
  group_by(sentiment, Speaker) %>% 
  count(word, sentiment, sort = T) %>% 
  top_n(10)

#get the sentiments score for clinton and trump participant
#trump
debate %>% 
  filter(Speaker %in% c("Trump","Clinton")) %>% 
  unnest_tokens(word, Text) %>% 
  anti_join(stop_words) %>% 
  inner_join(get_sentiments("bing")) %>% 
  count(Speaker, index = Line, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>% 
  ggplot(.,aes(index, sentiment, fill = Speaker)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Speaker, ncol = 10, scales = "free_x")
