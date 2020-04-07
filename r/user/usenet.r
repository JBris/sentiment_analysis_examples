# Source: https://www.tidytextmining.com/usenet.html
# Author: https://www.tidytextmining.com/preface.html#acknowledgements
# Data: http://qwone.com/~jason/20Newsgroups/

library(dplyr)
library(tidyr)
library(purrr)
library(readr)

training_folder <- "~/data/20news-bydate/20news-bydate-train/"

# Load data
# Define a function to read all files from a folder into a data frame
read_folder <- function(infolder) {
  tibble(file = dir(infolder, full.names = TRUE)) %>%
    mutate(text = map(file, read_lines)) %>%
    transmute(id = basename(file), text) %>%
    unnest(text)
}

# Use unnest() and map() to apply read_folder to each subfolder
raw_text <- tibble(folder = dir(training_folder, full.names = TRUE)) %>%
  mutate(folder_out = map(folder, read_folder)) %>%
  unnest(cols = c(folder_out)) %>%
  transmute(newsgroup = basename(folder), id, text)

raw_text