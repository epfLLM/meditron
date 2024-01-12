# Inference & Evaluation Instructions

## Main Pipeline

Run `inference_pipeline.sh` to execute the inference-evaluation pipeline.

The script first calls `inference.py` for inference. We use [vLLM](https://github.com/vllm-project/vllm) as the inference engine for generation. Model outputs are saved into the `benchmarks/generations` folder. Next the script calls `evaluate.py` to evaluate the model outputs against gold outputs.

## Run inference_pipeline.sh with parameters

    ./inference_pipeline.sh
        -c <model_name> or <checkpoint_name>
        -b <benchmark name>
        -s <num_shots> (in-context learning shots number, default is zero)
        -r 0 (cot prompting; 0: disabled, 1: enabled)
        -e vllm (inference backend, currently support vllm only)
        -m 0 (in-context learning with multiple seeds; 0: disabled, 1: enabled)
        -t 0 (self-consistency cot prompting; 0: disabled, 1: enabled)
        -d 32 (setting batch_size 32 for inference per gpu

Example for in-context learning (5-shot, multiple-seeds) with `pubmedqa` and `meditron-70b`:

    bash inference_pipeline.sh \
        -b pubmedqa
        -c meditron-70b \
        -s 5
        -m 1

Example for self-consistency cot prompting with `medqa` and `meditron-70b-cotmedqa`:

    bash inference_pipeline.sh \
        -b medqa
        -c meditron-70b-cotmedqa \
        -r 1
        -t 1

## Benchmarks

Add benchmarks to `benchmark.py` following an existing benchmark class. Currently require the benchmark is registered in the Huggingface dataset hub.

Example benchmark class:

```python
class MyBenchmark(Benchmark):
    '''
    MyBenchmark is <Your Description>

    Huggingface card: https://huggingface.co/datasets/<MyBenchmark>
    '''
    def __init__(self, name='pubmedqa') -> None:
        super().__init__(name)
        self.hub_name = "<MyBenchmark>"
        self.dir_name = '<Directory for MyBenchmark>'
        self.path = os.path.join(ROOT_DIR, 'benchmarks', 'datasets', self.dir_name)
        self.splits = ['train', 'validation', 'test']
        self.subsets = ['<If subset exist, specify here>']

    @staticmethod
    def custom_preprocessing(row):
        '''Add your custom preprocessing code here'''
        return row
```

## Models

Add models with associated local path or Huggingface repo name to `checkpoints` in `inference_pipeline.sh`.

For example:

    ["med42"]="m42-health/med42-70b" # Huggingface repo name
    ["meditron-70b"]="$meditron-70b/hf_checkpoints/raw/iter_23000/ # local path

Checkpoints from [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM) needs to be converted to Huggingface format in order to continue the inference & evaluation steps. To convert the checkpoints, please use the script provided by Megatron-LLM.

Here is an example for running the converstion script:
Specify proper parameters:

    NUM_IN_SHARDS=8 # number of input model shards
    NUM_OUT_SHARDS=8 # number of output model shards
    INPUT_DIR=<path to your Megatron checkpoint>
    OUTPUT_DIR=<path to save your HF model weights>
    UNSHARDED_DIR=<path for stroing unsharded Megatron checkpoint, temporary>

Execute the scripts

    ./megatron2hf.sh
