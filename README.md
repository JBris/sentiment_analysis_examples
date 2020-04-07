# sentiment_analysis_examples

## Table of Contents  

* [Introduction](#introduction)<a name="introduction"/>
* [Python](#python)<a name="python"/>
* [R](#r)<a name="r"/>

### Introduction

The Sentiment Analysis Examples repo contains several examples of sentiment analysis algorithms for use with textual data.

Examples can be found in the [python](python) directory.

If you're using Docker, execute [build.sh](build.sh) to get started.

### Python

Examples are typically written in python. From the [.env.example file](.env.example), you can see that scripts are written in python 3.8.2. A list of module dependencies can be found in the [Dockerfile](python/Dockerfile) and [requirements.txt](python/requirements.txt). You aren't forced to use Docker, and can use something like Conda instead if that's your preference.

If you opt to use Docker, you can view the [Makefile](Makefile) for relevant Docker commands. The `make penter` command will create a new container and execute the python CLI. The `make prun` command will run a python script. For example, `make prun d=imdb_reviews s=explore` will run [imdb_reviews/explore.py](python/imdb_reviews/explore.py)

Example sentiment analysis algorithms can be found in the [python](python) directory, and each example directory employs a similar structure. Python scripts will list any recommended article references and data sets. Download the recommended data sets and place them in the local data directory.

You can then execute various python scripts to analyze and model the data. It's recommended that you run explore.py then view.py first to better understand the distribution of the data.

### R


Additional examples are written in R. From the [.env.example file](.env.example), you can see that R scripts are written in version 3.6.3. A list of additional R and Python packages can be found in the [Dockerfile](r/Dockerfile). 

As the [docker-compose.yml](docker-compose.yml) file shows, this repo employs the [rocker/tidyverse image](https://hub.docker.com/r/rocker/tidyverse) which already includes the tidyverse collection and RStudio server.  

If you opt to use Docker, you can view the [Makefile](Makefile) for relevant Docker commands. The `make renter` command will allow users to execute shell commands within the R container. The `make rrun` command will run an R script. For example, `make rrun s=jane_austen` will run [$R_STUDIO_USER/jane_austen.r](r/user/jane_austen.r)

Example sentiment analysis algorithms can be found in the [r directory](r). You can execute various r scripts to analyze and model the data. 