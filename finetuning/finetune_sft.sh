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
INSTRUCT=0
CHECKPOINT_PATH=none
DATA=none
WANDB_PROJ=none
WANDB_ID=none
WANDB_ENTITY=none
ITERS=1000
SEQ_LEN=none
DATA_PATH=none
TRAINED_PATH=none
VAL_PATH=none
USR_LR=none
USR_MIN_LR=none
LOSS_MASK=0.0
SAVE_INTERVAL=800
IT=none
HELP_STR="[--rank=$RANK] [--size=$SIZE] [--tp=$TP] [--pp=$PP] [--gpus=$GPUS_PER_NODE] \
[--micro-batch=$MICRO_BATCH] [--global-batch=$GLOBAL_BATCH] [--nodes=$N_NODES] \
[--addr=$ADDR] [--wandb] [--instruct] [--checkpoint=...] [--data=...] [--iters=$ITERS] \
[--wandb-proj=none] [--wandb-id=none] [--wandb-entity=none] [--seq-len=...] \
[--val-path=none] [--out=...] [--lr=lr minlr] [--loss-mask=$LOSS_MASK]
[--save-interval=$SAVE_INTERVAL] [--it=$IT] [--help]"


# define help function
help () {
	echo "Usage: $0 <gpt/llama/llama2/codellama/falcon> $HELP_STR"
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
		--wandb-project) WANDB_PROJ=$2; shift; shift;;
		--wandb-id) WANDB_ID=$2; shift; shift;;
		--wandb-entity) WANDB_ENTITY=$2; shift; shift;;
		--instruct) INSTRUCT=1; shift;;
		--checkpoint) CHECKPOINT_PATH=$2; shift; shift;;
		--data) DATA_PATH=$2; shift; shift;;
		--iters) ITERS=$2; shift; shift;;
		--seq-len) SEQ_LEN=$2; shift; shift;;
		--out) TRAINED_PATH=$2; shift; shift;;
		--val-path) VAL_PATH=$2; shift; shift;;
		--lr) USR_LR=$2; USR_MIN_LR=$3; shift; shift; shift;;
		--loss-mask) LOSS_MASK=$2; shift; shift;;
		--save-interval) SAVE_INTERVAL=$2; shift; shift;;
		--it) IT=$2; shift; shift;;
		*) echo unknown argument $1; help; exit 1;;
	esac
done


# set args
if [[ $CHECKPOINT_PATH = none ]]; then
	CHECKPOINT_PATH=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}-${SIZE}b-tp$TP-pp$PP
fi

if [[ $INSTRUCT = 1 ]]; then
	LR="2e-5"
	MIN_LR="2e-6"
	if [[ $TRAINED_PATH = none ]]; then
		TRAINED_PATH=$CHECKPOINT_PATH-instructed
	fi
else
	LR="3e-4"
	MIN_LR="3e-4"
	if [[ $TRAINED_PATH = none ]]; then
		TRAINED_PATH=$CHECKPOINT_PATH-pretrained
	fi
fi

TENSORBOARD_PATH=$TRAINED_PATH/logging
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $N_NODES --node_rank
                  $RANK --master_addr $ADDR --master_port 6020"

if [[ $MODEL = falcon ]]; then
	if [[ $DATA_PATH = none ]]; then
		DATA_PATH=/pure-mlo-scratch/pagliard/data/wikitext-falcon/wiki-train_text_document
	fi
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--parallel_attn"
	if [[ $SEQ_LEN = none ]]; then
		SEQ_LEN=2048
	fi
elif [[ $MODEL = llama ]] || [[ $MODEL = llama2 ]] || [[ $MODEL = codellama ]]; then
	EXTRA_IDS="[bib_ref],[/bib_ref],[fig_ref],[/fig_ref],[bib],[/bib],[fig],[/fig],[table],[/table],[formula],[/formula]"
	EXTRA_ARGS="--vocab_file=/pure-mlo-scratch/llama/tokenizer.model --use_rms_norm
	            --glu_activation swiglu --no_tie_embed_logits"
	if [[ $INSTRUCT = 1 ]]; then
		if [[ $DATA_PATH = none ]]; then
			DATA_PATH=/pure-mlo-scratch/alhernan/data/orca/orca
		fi
		EXTRA_IDS="$EXTRA_IDS,<|im_start|>,<|im_end|>"
	else
		if [[ $DATA_PATH = none ]]; then
			DATA_PATH=/pure-mlo-scratch/data/tokenized/pubmed-all/pubmed-all-llama_text_document
		fi
	fi
	TOKENIZER=SentencePieceTokenizer
	EXTRA_ARGS="$EXTRA_ARGS --vocab_extra_ids_list $EXTRA_IDS"
	if [[ $MODEL == llama ]]; then
		if [[ $SEQ_LEN = none ]]; then
			SEQ_LEN=2048
		fi
		EXTRA_ARGS="$EXTRA_ARGS --vocab_file=/pure-mlo-scratch/llama2/Llama-2-7b-hf/tokenizer.model"
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-6"
	elif [[ $MODEL == llama2 ]]; then
		if [[ $SEQ_LEN = none ]]; then
			SEQ_LEN=4096
		fi
		EXTRA_ARGS="$EXTRA_ARGS --vocab_file=/pure-mlo-scratch/llama2/Llama-2-7b-hf/tokenizer.model"
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-5"
		if (( $SIZE > 13 )); then  # llama 2, 34B and 70B
			LR="1.5e-4"
			MIN_LR="1.5e-5"
		fi
	else  # codellama
		if [[ $SEQ_LEN = none ]]; then
			SEQ_LEN=16384
		fi
		EXTRA_ARGS="$EXTRA_ARGS --vocab_file=/pure-mlo-scratch/codellama/CodeLlama-7b/tokenizer.model --rope_theta 1e6"
	fi
elif [[ $MODEL = gpt ]]; then
	if [[ $DATA_PATH = none ]]; then
		DATA_PATH=/scratch/wikitext-megatron/wikitext-train_text_document
	fi
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--num_layers 4 --hidden_size 512 --num_attention_heads 8"
	if [[ $SEQ_LEN = none ]]; then
		SEQ_LEN=2048
	fi
else
	echo "Model should be either gpt, llama or falcon, not $MODEL"
	help
	exit 1
fi
COMMON_ARGS="--use_flash_attn --no_bias_gelu_fusion
	     --seq_length $SEQ_LEN --max_position_embeddings $SEQ_LEN
             --log_interval 1 --save_interval $SAVE_INTERVAL --eval_interval 100
             --eval_iters 10 --hidden_dropout 0.0 --position_embedding_type rotary
	     --no_bias_dropout_fusion --use_checkpoint_args
	     --attention_dropout 0.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5
	     --lr_decay_style cosine --lr_warmup_fraction 0.1 --lr $LR --min_lr $MIN_LR
	     --weight_decay 0.1 --sequence_parallel --recompute_granularity selective
	     --log_timers_to_tensorboard --scalar_loss_mask=$LOSS_MASK
	     --rope_scaling_factor 1.0"

if [[ $INSTRUCT = 1 ]]; then
	COMMON_ARGS="$COMMON_ARGS --variable_seq_lengths --data_type instruction --metrics all"
	if [[ $CHECKPOINT_PATH != $TRAINED_PATH ]]; then
		COMMON_ARGS="$COMMON_ARGS --finetune"
	fi
else
	COMMON_ARGS="$COMMON_ARGS --metrics perplexity accuracy count_loss_mask"
fi

if [[ $CHECKPOINT_PATH != $TRAINED_PATH ]]; then
 	COMMON_ARGS="$COMMON_ARGS --train_iters $ITERS"
fi

if [[ $WANDB = 1 ]]; then
	COMMON_ARGS="$COMMON_ARGS --wandb_logger"
	if [[ $WANDB_PROJ != none ]]; then
		COMMON_ARGS="$COMMON_ARGS --wandb_project $WANDB_PROJ"
	fi
	if [[ $WANDB_ID != none ]]; then
		COMMON_ARGS="$COMMON_ARGS --wandb_id $WANDB_ID"
	fi
	if [[ $WANDB_ENTITY != none ]]; then
		COMMON_ARGS="$COMMON_ARGS --wandb_entity $WANDB_ENTITY"
	fi
fi

if [[ $VAL_PATH = none ]]; then
	DATA_ARGS="--data_path $DATA_PATH"
else
	DATA_ARGS="--train_data_path $DATA_PATH --valid_data_path $VAL_PATH"
fi

if [[ $IT != none ]]; then
	COMMON_ARGS="$COMMON_ARGS --load_iteration $IT"
fi

# print some args
echo
echo Settings:
echo RANK=$RANK
echo ADDR=$ADDR
echo N_NODES=$N_NODES
echo DATA_ARGS=$DATA_ARGS
echo CHECKPOINT_PATH=$CHECKPOINT_PATH
echo TRAINED_PATH=$TRAINED_PATH
echo MODEL=$MODEL
echo TP=$TP
echo PP=$PP
echo MICRO_BATCH=$MICRO_BATCH
echo GLOBAL_BATCH=$GLOBAL_BATCH
echo INSTRUCT=$INSTRUCT
echo COMMON_ARGS=$COMMON_ARGS
echo EXTRA_ARGS=$EXTRA_ARGS
echo


# finally, call finetune.py, make sure to specify the path to your Megatron-LLM directory
CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=16 torchrun $DISTRIBUTED_ARGS Megatron-LLM/finetune.py \
       --tensor_model_parallel_size $TP \
       --pipeline_model_parallel_size $PP  \
       --load $CHECKPOINT_PATH \
       --save $TRAINED_PATH \
       --tensorboard_dir $TENSORBOARD_PATH \
       $DATA_ARGS \
       --model_name $MODEL \
       --tokenizer_type $TOKENIZER \
       --bf16 \
       --global_batch_size $GLOBAL_BATCH \
       --micro_batch_size $MICRO_BATCH \
       --num_workers=2 \
       $EXTRA_ARGS \
       $COMMON_ARGS
