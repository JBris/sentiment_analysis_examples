#!/usr/bin/env bash 

#See https://raw.githubusercontent.com/aaronkub/machine-learning-examples/master/imdb-sentiment-analysis/preprocess_reviews.sh

DIR="$(pwd)/data/aclImdb"
FULL_DIR="${DIR}/movie_data"
mkdir -p "$FULL_DIR"
current_dir=$(pwd)
cd "$DIR"

# puts four files in the combined_files directory:
# full_train.txt, full_test.txt, original_train_ratings.txt, and original_test_ratings.txt
for split in train test
do
    ( for sentiment in pos neg
    do 
    
        [[ ! "$(ls -A ${split}/${sentiment} )" ]] && continue

        for file in $split/$sentiment/*; 
        do
            (cat $file; echo) >> movie_data/full_${split}.txt
            echo "Added $file"
	        # This line adds files containing the original reviews if desired
            # echo $file | cut -d '_' -f 2 | cut -d "." -f 1 >> combined_files/original_${split}_ratings.txt; 
        done 
    done ) &
done
wait

cd "$current_dir"
