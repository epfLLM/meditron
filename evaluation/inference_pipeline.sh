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
BENCHMARK=medmcqa
SHOTS=0
COT=0
SC_COT=0
MULTI_SEED=0
BACKEND=vllm
WANDB=1
BATCH_SIZE=16

HELP_STR="[--checkpoint=$CHECKPOINT_NAME] [--benchmark=$BENCHMARK] [--help]"

help () {
	echo "Usage: $0 <vllm> $HELP_STR"
}

if [[ $# = 1 ]] && [[ $1 = "-h" ]] || [[ $1 = "--help" ]]; then
	help
	exit 0
elif [[ $# = 0 ]]; then
	help
	exit 1
fi

while getopts c:b:s:r:e:m:t:d: flag
do
    case "${flag}" in
        c) CHECKPOINT_NAME=${OPTARG};;
        b) BENCHMARK=${OPTARG};;
        s) SHOTS=${OPTARG};;
        r) COT=${OPTARG};;
        e) BACKEND=${OPTARG};;
        m) MULTI_SEED=${OPTARG};;
        t) SC_COT=${OPTARG};;
        d) BATCH_SIZE=${OPTARG};;
    esac
done

CHECKPOINT=${checkpoints[$CHECKPOINT_NAME]}

echo
echo "Running inference pipeline"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Benchmark: $BENCHMARK"
echo "Backend: $BACKEND"
echo "Shots: $SHOTS"
echo "COT: $COT"
echo "Multi seed: $MULTI_SEED"
echo "SC COT: $SC_COT"
echo "BATCH_SIZE: $BATCH_SIZE"
echo

COMMON_ARGS="--checkpoint  $CHECKPOINT \
    --checkpoint_name ${CHECKPOINT_NAME} \
    --benchmark $BENCHMARK \
    --shots $SHOTS \
    --batch_size $BATCH_SIZE"
ACC_ARGS="--checkpoint $CHECKPOINT_NAME \
    --benchmark $BENCHMARK \
    --shots $SHOTS"

if [[ $COT = 1 ]]; then
    echo "COT Prompting"
   COMMON_ARGS="$COMMON_ARGS --cot"
fi

if [[ $MULTI_SEED = 1 ]]; then
    echo "In-context with Multi Seed"
    COMMON_ARGS="$COMMON_ARGS --multi_seed"
    ACC_ARGS="$ACC_ARGS --multi_seed"
fi

if [[ $SC_COT = 1 ]]; then
    echo "SC-COT Prompting"
    COMMON_ARGS="$COMMON_ARGS --sc_cot"
    ACC_ARGS="$ACC_ARGS --sc_cot"
fi

if [[ $WANDB = 1 ]]; then
    echo "WANDB Log Enabled"
    ACC_ARGS="$ACC_ARGS --wandb"
fi

echo inference.py $COMMON_ARGS
python inference.py $COMMON_ARGS
python evaluate.py $ACC_ARGS
