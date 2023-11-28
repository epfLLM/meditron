#!/bin/bash

"""
This script downloads then cleans the guidelines corpus.

Important note to users:
  Scraping logic will inevitably rot over time, and probably quickly.

  These are best-effort contemporary (November 2023) reconstructions of our original data
  collection effort, which took place some months before. As you can see in places
  the logic is fairly hacky.

  We will support interested users in the immediate period after the code
  release, but it's impossible to imagine supporting the scraping logic
  beyond that.

  Best Wishes,

  Antoine, Alexandre, and Kyle
"""

# NOTE: If you run scrapers outside of guidelines/download.sh, you have to start the GROBID service beforehand:
# ./guidelines/serve_grobid.sh

PATH_TO_SCRAPERS='/scrapers'               # Path to scrapers directory
PATH_TO_RAW=$PATH_TO_SCRAPERS"/raw"        # Raw scraped guidelines directory
PATH_TO_CLEAN=$PATH_TO_SCRAPERS"/clean"    # Clean guidelines directory


echo "1. Scraping guidelines..."

echo "1. a) Running 12/16 Chrome-based scrapers..."
# Downloads guidelines from each source to {PATH_TO_SCRAPERS}/raw/{source}.jsonl
# You can download specific sources by adding them to the --sources flag
python scrapers/scrapers.py \
    --path $PATH_TO_SCRAPERS \
#    --sources aafp cco cdc cma cps drugs guidelinecentral icrc idsa magic spor who

echo "\n1. b) Running 4/16 Typescript-based scrapers..."
TS_SCRAPERS=("mayo" "nice" "rch" "wikidoc")

# Loop through each scraper directory
for TS_SCRAPER_DIR in "${TS_SCRAPERS[@]}"; do
    echo "Running scraper in $TS_SCRAPER_DIR..."
    cd "scrapers/$TS_SCRAPER_DIR"

    # Install dependencies
    if npm install --silent; then

        # Compile and run the scraper
        if tsc && node js/index.js; then
            echo "Scraper in $TS_SCRAPER_DIR completed successfully."
        else
            echo "Error: Failed to run scraper in $TS_SCRAPER_DIR. This might be due to website updates. Skipping..."
            cd ../..
            continue
        fi
    else
        echo "Error: Failed to install dependencies for scraper in $TS_SCRAPER_DIR. Skipping..."
        cd ../..
        continue
    fi
    
    cd ../..
done


# 2. Clean guidelines
echo "2. Cleaning guidelines..."
python guidelines/clean.py \
    --process \
    --raw_dir $PATH_TO_RAW \
    --save_dir $PATH_TO_CLEAN
    
# 3. Combine guidelines into guidelines.jsonl, add IDs, split into train/val/test
echo "3. Combining guidelines..."
python guidelines/clean.py \
    --save_dir $PATH_TO_CLEAN \
