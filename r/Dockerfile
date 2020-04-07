ARG TIDYVERSE_TAG

FROM rocker/tidyverse:$TIDYVERSE_TAG

ENV R_PACKAGES="\
    tidytext \
    janeaustenr \
    reshape2 \
    wordcloud \
    rtweet \
    textdata \
    sentimentr \
    ndjson \
" 

RUN install2.r --error $R_PACKAGES

#RUN install2.r --error --deps TRUE $R_PACKAGES
