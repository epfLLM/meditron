#!/bin/bash

# Tokenize GAP-Replay (Train + Test)
# You will need to clone the Megatron-LLM repository https://github.com/epfLLM/Megatron-LLM
# and provide a valid path to it

echo "Tokenize GAP-Replay (Train + Test)"

TRAINER_PATH = /model-parallel-trainer
echo "Trainer path: $TRAINER_PATH"

DATA_PATH = /data/gap-replay
echo "GAP-Replay path: $GAP_REPLAY_PATH"

python $TRAINER_PATH/tools/preprocess_data.py \
       --input $DATA_PATH/gap_replay_train.jsonl \
       --output_prefix /path/to/output/dir \
       --vocab_file /path/to/tokenizer.model \
       --dataset_impl mmap \
       --tokenizer_type SentencePieceTokenizer \
       --vocab_extra_ids_list "[bib_ref],[/bib_ref],[fig_ref],[/fig_ref],[bib],[/bib],[fig],[/fig],[table],[/table],[formula],[/formula]" \
       --json_keys "text" \
       --workers 8 \
       --chunk_size 2048 \
       --append_eod

python $TRAINER_PATH/tools/preprocess_data.py \
       --input $DATA_PATH/gap_replay_test.jsonl \
       --output_prefix /path/to/output/dir \
       --vocab_file /path/to/tokenizer.model \
       --dataset_impl mmap \
       --tokenizer_type SentencePieceTokenizer \
       --vocab_extra_ids_list "[bib_ref],[/bib_ref],[fig_ref],[/fig_ref],[bib],[/bib],[fig],[/fig],[table],[/table],[formula],[/formula]" \
       --json_keys "text" \
       --workers 8 \
       --chunk_size 2048 \
       --append_eod

