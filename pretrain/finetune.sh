#! /bin/bash

# default arguments
SIZE=7
TP=8
PP=1
GPUS_PER_NODE=8
MICRO_BATCH=1
GLOBAL_BATCH=12
RANK=0
N_NODES=1
ADDR=localhost
WANDB=0
DO_EVAL=0
TEST=uptodate
ITER=0
EXP_NAME=guidelines-70b


HELP_STR="[--rank=$RANK] [--size=$SIZE] [--tp=$TP] [--pp=$PP] [--gpus=$GPUS_PER_NODE] \
[--micro-batch=$MICRO_BATCH] [--global-batch=$GLOBAL_BATCH] [--nodes=$N_NODES] \
[--addr=$ADDR] [--wandb] [--test=$TEST] [--iter=$ITER] [--exp=$EXP_NAME] \
[--do_eval=$DO_EVAL] [--help]"


# define help function
help () {
	echo "Usage: $0 <gpt/llama/llama2/falcon> $HELP_STR"
}

# parse arguments, three modes
# mode1 = -h or --help requested
if [[ $# = 1 ]] && [[ $1 = "-h" ]] || [[ $1 = "--help" ]]; then
	help
	exit 0
# mode2 = not arguments given
elif [[ $# = 0 ]]; then
	help
	exit 1
fi
# mode3 = correct usage, read model
MODEL=$1
shift
while [[ $# -gt 0 ]]; do
	case $1 in
		--tp) TP="$2"; shift; shift;;
		--pp) PP="$2"; shift; shift;;
		--size) SIZE="$2"; shift; shift;;
		--gpus) GPUS_PER_NODE="$2"; shift; shift;;
		--micro-batch) MICRO_BATCH="$2"; shift; shift;;
		--global-batch) GLOBAL_BATCH="$2"; shift; shift;;
		--rank) RANK=$2; shift; shift;;
		--nodes) N_NODES=$2; shift; shift;;
		--addr) ADDR=$2; shift; shift;;
		--wandb) WANDB=1; shift;;
		--test) TEST=$2; shift; shift;;
		--iter) ITER=$2; shift; shift;;
		--exp) EXP_NAME=$2; shift; shift;;
		--do_eval) DO_EVAL=1; shift;;
		*) echo unknown argument $1; help; exit 1;;
	esac
done


# set args
LR="3e-4"
MODEL_CONFIG=${MODEL}-${SIZE}b-tp$TP-pp$PP
EVAL_ITERS=20

if [[ $EXP_NAME = true_baseline ]]; then
	LOAD_CHECKPOINT_PATH=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/$MODEL_CONFIG
	if [[ $DO_EVAL = 0 ]]; then
		echo "Don't train the true_baseline lol"
		exit 1
	fi
else
	LOAD_CHECKPOINT_PATH=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL_CONFIG}
	SAVE_CHECKPOINT_PATH=/pure-mlo-scratch/trial-runs/${EXP_NAME}/checkpoints/${MODEL_CONFIG}
	TENSORBOARD_PATH=/pure-mlo-scratch/trial-runs/${EXP_NAME}/tensorboards/${MODEL_CONFIG}
fi

if [[ $ITER > 0 ]]; then
	echo $ITER  > $LOAD_CHECKPOINT_PATH/latest_checkpointed_iteration.txt
	LOAD_CHECKPOINT_PATH=/pure-mlo-scratch/trial-runs/${EXP_NAME}/checkpoints/tmp/
fi

if [[ $DO_EVAL = 1 ]]; then
	SAVE_CHECKPOINT_PATH=/pure-mlo-scratch/trial-runs/debug/checkpoints/${MODEL_CONFIG}
	TENSORBOARD_PATH=/pure-mlo-scratch/trial-runs/debug/tensorboards/${MODEL_CONFIG}
	COMMON_ARGS="--finetune "
fi

TRAIN_DATA_PATH=/pure-mlo-scratch/data/tokenized/GAP-replay-train/GAP-replay-train_text_document
VALID_DATA_PATH=/pure-mlo-scratch/data/tokenized/GAP-validation/GAP-validation_text_document

# EVAL_ITERS is computed after determining SEQ_LEN
if [[ $TEST = pubmed ]]; then
	TEST_DATA_PATH=/pure-mlo-scratch/data/tokenized/pubmed-all-validation/pubmed-all-validation-llama_text_document
	# EVAL_ITERS=100
	echo unknown number of tokens
	exit 1
elif [[ $TEST = uptodate ]]; then
	TEST_DATA_PATH=/pure-mlo-scratch/data/tokenized/uptodate-only/uptodate-llama_text_document
	# EVAL_ITERS=18
	EVAL_TOKENS=36698354
elif [[ $TEST = replay ]]; then
	TEST_DATA_PATH=/pure-mlo-scratch/data/tokenized/replay-only/valid-replay-1B-llama_text_document
	# EVAL_ITERS=100
	echo unknown number of tokens
	exit 1
elif [[ $TEST = gap ]]; then
	TEST_DATA_PATH=/pure-mlo-scratch/data/tokenized/GAP-validation/GAP-validation_text_document
	EVAL_TOKENS=1405318722
else
	echo "Test should be either pubmed, uptodate or replay, not $TEST"
	help
	exit 1
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $N_NODES --node_rank
                $RANK --master_addr $ADDR --master_port 6001"

if [[ $MODEL = falcon ]]; then
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--use_multiquery_attn --parallel_attn"
	SEQ_LEN=2048
elif [[ $MODEL = llama ]] || [[ $MODEL = llama2 ]]; then
	TOKENIZER=SentencePieceTokenizer
	EXTRA_ARGS='--vocab_file=/pure-mlo-scratch/llama/tokenizer.model --use_rms_norm
	            --glu_activation swiglu --no_tie_embed_logits
		    	--vocab_extra_ids_list "[bib_ref],[/bib_ref],[fig_ref],[/fig_ref],[bib],[/bib],[fig],[/fig],[table],[/table],[formula],[/formula]"'
	if [[ $MODEL == llama ]]; then
		SEQ_LEN=2048
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-6"
	else  # llama 2
		SEQ_LEN=4096
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-5"
		if (( $SIZE > 13 )); then  # llama 2, 34B and 70B
			LR="1.5e-4"
		fi
	fi
elif [[ $MODEL = gpt ]]; then
	DATA_PATH=/scratch/wikitext-megatron/wikitext-train_text_document
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--num_layers 4 --hidden_size 512 --num_attention_heads 8"
	SEQ_LEN=2048
else
	echo "Model should be either gpt, llama or falcon, not $MODEL"
	help
	exit 1
fi

# this is equal to ceil(eval_tokens/seq_len/batch)
EVAL_ITERS=$(((EVAL_TOKENS + SEQ_LEN*GLOBAL_BATCH)/(SEQ_LEN*GLOBAL_BATCH)))

COMMON_ARGS="$COMMON_ARGS --use_flash_attn --no_bias_gelu_fusion
		--seq_length $SEQ_LEN --max_position_embeddings $SEQ_LEN
		--log_interval 1 --eval_interval 10 --save_interval 20
		--use_checkpoint_args --hidden_dropout 0.0
		--position_embedding_type rotary
		--no_bias_dropout_fusion --attention_dropout 0.0
		--adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5
		--lr_decay_style cosine --lr_warmup_iters 5 --lr $LR --min_lr 1e-6
		--weight_decay 0.1 --sequence_parallel --recompute_granularity selective
		--log_validation_ppl_to_tensorboard
        --log_memory_to_tensorboard
        --log_timers_to_tensorboard
		--num_workers 0 --dataloader_type cyclic
		--train_data_path $TRAIN_DATA_PATH
		--valid_data_path $VALID_DATA_PATH
		--test_data_path $TEST_DATA_PATH
		--eval_iters $EVAL_ITERS
		--train_iters 102"

if [[ $DO_EVAL = 1 ]]; then
	COMMON_ARGS="$COMMON_ARGS --eval_only --use_checkpoint_opt_param_scheduler"
fi

if [[ $WANDB = 1 ]]; then
	COMMON_ARGS="$COMMON_ARGS --wandb_logger --wandb_project <YOUR W&B PROJECT> --wandb_name $EXP_NAME"
fi

echo
echo Settings:
echo RANK=$RANK
echo ADDR=$ADDR
echo N_NODES=$N_NODES
echo DATA_PATH=$DATA_PATH
echo CHECKPOINT_PATH=$LOAD_CHECKPOINT_PATH
echo MODEL=$MODEL
echo TP=$TP
echo PP=$PP
echo MICRO_BATCH=$MICRO_BATCH
echo GLOBAL_BATCH=$GLOBAL_BATCH
echo EVAL_ITERS=$EVAL_ITERS
echo

CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=16 torchrun $DISTRIBUTED_ARGS Megatron-LLM/finetune.py \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP  \
	--load $LOAD_CHECKPOINT_PATH \
	--save $SAVE_CHECKPOINT_PATH \
	--tensorboard_dir $TENSORBOARD_PATH \
	--model_name $MODEL \
	--tokenizer_type $TOKENIZER \
	--bf16 \
	--global_batch_size $GLOBAL_BATCH \
	--micro_batch_size $MICRO_BATCH \
	$EXTRA_ARGS \
	$COMMON_ARGS
