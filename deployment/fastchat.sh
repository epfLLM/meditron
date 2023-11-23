#!/bin/bash

declare -A checkpoints

CHECKPOINT_DIR="/pure-mlo-scratch/trial-runs/"

checkpoints=(["mpt"]="mosaicml/mpt-7b" \
             ["falcon"]="tiiuae/falcon-7b" \
             ["mistral"]="mistralai/Mistral-7B-Instruct-v0.1" \
             ["zephyr"]="HuggingFaceH4/zephyr-7b-beta" \
             ["baseline-7b"]="/pure-mlo-scratch/llama2/converted_HF_7B_8shard/" \
             ["pmc-7b"]="/pure-mlo-scratch/trial-runs/pmc-7b/hf_checkpoints/raw/pmc-llama-7b" \
             ["meditron-7b"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/raw/release/" \
             ["clinical-camel"]="wanglab/ClinicalCamel-70B" \
             ["med42"]="m42-health/med42-70b" \

             ["baseline-70b"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/raw/release/" \
             ["meditron-70b"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/raw/iter_23000/" \

             ["baseline-medmcqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/medmcqa/" \
             ["baseline-pubmedqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/pubmedqa/" \
             ["baseline-medqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/medqa/" \
             ["baseline-cotmedmcqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/cotmedmcqa/" \
             ["baseline-cotpubmedqa"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/cotpubmedqa/" \
             ["baseline-medical"]="${CHECKPOINT_DIR}baseline-7b/hf_checkpoints/instruct/medical/" \

             ["pmc-medmcqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/medmcqa/" \
             ["pmc-medqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/medqa-32/" \
             ["pmc-pubmedqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/pubmedqa/" \
             ["pmc-cotpubmedqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/cotpubmedqa/" \
             ["pmc-cotmedmcqa"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/cotmedmcqa/"\
             ["pmc-medical"]="${CHECKPOINT_DIR}pmc-7b/hf_checkpoints/instruct/medical/"\

             ["meditron-7b-medmcqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/medmcqa/" \
             ["meditron-7b-pubmedqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/pubmedqa/" \
             ["meditron-7b-medqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/medqa/" \
             ["meditron-7b-cotpubmedqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/cotpubmedqa/" \
             ["meditron-7b-cotmedmcqa"]="${CHECKPOINT_DIR}meditron-7b/hf_checkpoints/instruct/cotmedmcqa/" \

             ["baseline-70b-medqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/medqa/" \
             ["baseline-70b-medmcqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/medmcqa/" \
             ["baseline-70b-pubmedqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/pubmedqa/" \
             ["baseline-70b-cotmedqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/cotmedqa/" \
             ["baseline-70b-cotmedmcqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/cotmedmcqa/" \
             ["baseline-70b-cotpubmedqa"]="${CHECKPOINT_DIR}baseline-70b/hf_checkpoints/instruct/cotpubmedqa/" \

             ["meditron-70b-medmcqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/medmcqa/" \
             ["meditron-70b-pubmedqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/pubmedqa" \
             ["meditron-70b-medqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/medqa/" \
             ["meditron-70b-cotmedmcqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/cotmedmcqa/" \
             ["meditron-70b-cotpubmedqa"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/cotpubmedqa" \
             ["meditron-70b-cotmedqa-qbank"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/cotmedqa/" \
             ["meditron-70b-instruct"]="${CHECKPOINT_DIR}meditron-70b/hf_checkpoints/instruct/medical")

CHECKPOINT_NAME=meditron-70b
NUM_GPU=8
MODE="single" # [single, multi1, multi2]

HELP_STR="[--checkpoint=$CHECKPOINT_NAME] [--num_gpu=$NUM_GPU] [--mode=$MODE] [--help]"
# define help function
help () {
	echo "Usage: $0 <gpt/llama/llama2/falcon> $HELP_STR"
}

if [[ $# = 1 ]] && [[ $1 = "-h" ]] || [[ $1 = "--help" ]]; then
	help
	exit 0
elif [[ $# = 0 ]]; then
	help
	exit 1
fi

while getopts c:n:m: flag
do
    case "${flag}" in
        c) CHECKPOINT_NAME=${OPTARG};;
        n) NUM_GPU=${OPTARG};;
        m) MODE=${OPTARG};;
    esac
done

CHECKPOINT=${checkpoints[$CHECKPOINT_NAME]}

echo
echo "Running chat model worker"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Number of GPUs: $NUM_GPU"
echo "Mode: $MODE"
echo

if [ "$MODE" = "single" ]; then
    echo "Running single fastchat vllm worker"
    python -m fastchat.serve.vllm_worker \
        --model-path $CHECKPOINT \
        --model-name $CHECKPOINT_NAME \
        --conv-template one_shot_medical \
        --tensor-parallel-size $NUM_GPU \
        --controller-address http://localhost:21001 \
        --worker-address http://localhost:31000 \
        --port 31000 \
        --seed 0
    exit 0
elif [ "$MODE" = "multi1" ]; then
    echo "Running multiple fastchat vllm workers, worker 1"
    python -m fastchat.serve.vllm_worker \
        --model-path $CHECKPOINT \
        --model-name $CHECKPOINT_NAME \
        --num-gpus $NUM_GPU \
        --controller-address http://localhost:21001 \
        --port 31000 \
        --worker-address http://localhost:31000
    exit 0
elif [ "$MODE" = "multi2" ]; then
    echo "Running multiple fastchat vllm workers, worker 2"
    python -m fastchat.serve.vllm_worker \
        --model-path $CHECKPOINT \
        --model-name $CHECKPOINT_NAME \
        --num-gpus $NUM_GPU \
        --controller http://localhost:21001 \
        --port 31001 \
        --worker http://localhost:31001
    exit 0
fi
