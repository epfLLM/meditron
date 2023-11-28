#!/bin/bash

# PubMed Abstracts + Papers downloader

# Load data, pre-process, and tokenize the GAP-Replay dataset. 
# Comment out the steps you don't need and adjust the paths as desired.

# 1. Load data: S2ORC + Abstracts and join with metadata from Papers. After this command, 
# - Abstracts with metadata are stored in /data/abstracts-PubMed_metadata.jsonl 
# - Full-text articles with metadata are stored in /data/s2orc-PubMed_metadata.jsonl
# NOTE: You will need to provide your Semantic Scholar API key.  
python pubmed/load.py \
       --dataset all \
       --data_path $GAP_DATA_DIR \
       --key_path pubmed/keys.json 

# 2. Deduplication: remove abstracts for which we already have full-text articles
python pubmed/process.py \
       --deduplicate\
       --source_path /data/abstracts-PubMed_metadata.jsonl \
       --save_path /data/abstracts-PubMed_dedup.jsonl

# 3. Clean PubMed abstracts and full-text articles
python pubmed/process.py \
       --dataset s2orc \
       --source_path /data/s2orc-PubMed_metadata.jsonl \
       --save_path /data/s2orc-PubMed_processed.jsonl

python process.py \
       --dataset abstracts \
       --source_path /data/abstracts-PubMed_dedup.jsonl \
       --save_path /data/abstracts-PubMed_processed.jsonl

# 4. Train-test split for each dataset
python pubmed/process.py \
       --train_test_split \
       --source_path /data/s2orc-PubMed_processed.jsonl \
       --split_ratio 0.03 \

python pubmed/process.py \
       --train_test_split \
       --source_path /data/abstracts-PubMed_processed.jsonl \
       --split_ratio 0.03 \

python pubmed/process.py \
       --train_test_split \
       --source_path /data/guidelines.jsonl \
       --split_ratio 0.05 \
