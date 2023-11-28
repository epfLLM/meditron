#!/bin/bash

# Entire data pipeline for the GAP-Replay dataset. 

DATA_PATH=/data/gap-replay
echo "GAP-Replay path: $DATA_PATH"

echo "Downloading guidelines"
./guidelines/download.sh $DATA_PATH
echo "Done"

echo "Downloading Pubmed Papers + Abstracts"
./pubmed/download.sh $DATA_PATH
echo "Done"

echo "Downloading Replay corpus"
./replay/download.sh
echo "Done"

echo "Combining files into GAP-Replay (Train + Test)"
python pubmed/process.py \
       --combine \
       --source_path $DATA_PATH/s2orc-PubMed_processed_train.jsonl,\
       $DATA_PATH/abstracts-PubMed_processed_train.jsonl,\
       $DATA_PATH/guidelines_train.jsonl,\
       $DATA_PATH/replay.jsonl\
       --save_path $DATA_PATH/gap_replay_train.jsonl

python pubmed/process.py \
       --combine \
       --source_path $DATA_PATH/s2orc-PubMed_processed_test.jsonl,\
       $DATA_PATH/abstracts-PubMed_processed_test.jsonl,\
       $DATA_PATH/guidelines_test.jsonl,\
       --save_path $DATA_PATH/gap_replay_test.jsonl
