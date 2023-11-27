"""
This script implements the main pipeline of data preprocessing for fine-tuning.
"""

import argparse
import os
import sys
import sys
sys.path.append('../')
from benchmarks import benchmark_factory
from benchmarks import load_instruction


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


INSTRUCTIONS = {
    'medmcqa': {'partition': 'validation', 'instructions': 'medmcqa', 'cot_col': 'exp'},
    'pubmedqa': {'partition': 'validation', 'instructions': 'pubmedqa', 'cot_col': 'LONG_ANSWER'},
    'medqa': {'partition': 'test', 'instructions': 'medqa'},
    'medicationqa': {'partition': '', 'instructions': 'open_question'},
    'mmlu_medical': {'partition': 'test', 'instructions': 'medmcqa'},
    'mmlu_general': {'partition': 'test', 'instructions': 'medmcqa'},
    "gsm8k": {'partition': 'test', 'instructions': 'gsm8k', 'cot_col': 'steps'},
}

def benchmark_preparation(data_obj, partition, args):
    """
    Runs the benchmark preparation pipeline on a given benchmark.

    :param data_obj: benchmark.Benchmark, the benchmark to run the preparation pipeline on
    :param partition: str, the partition to run the preparation pipeline on
    :param args: argparse.Namespace, the arguments to run the preparation pipeline
    """
    # Load & preprocess [partition] data
    data_obj.load_data(partition=partition)
    data_obj.preprocessing(partition=partition)

    # Get instructions
    prompt_name = INSTRUCTIONS[args.benchmark]['instructions']
    if args.tags:
        prompt_name = prompt_name + '_tags'
    if args.cot:
        prompt_name = prompt_name + '_cot'
    print(f'Prompt used for evaluation: {prompt_name}')
    instruction = load_instruction(prompt_name)
    print(f'Instruction used for evaluation: \n\t{instruction["system"]}\n\t{instruction["user"]}\n')

    # Adding instruction
    if args.cot:
        data_obj.add_instruction(
            instruction=instruction,
            partition=partition,
            cot_column = INSTRUCTIONS[args.benchmark].get('cot_col', None))
    else:
        data_obj.add_instruction(
            instruction=instruction,
            partition=partition)
    return prompt_name

def main(args):
    """
    Runs the main evaluation pipeline on the given generations.

    :param benchmark: str, the name of the bechmark used for the evaluation
    :param distination_path: str, the name of the checkpoint that produced the generations.
    """
    # Instantiate benchmark and load data
    data_obj = benchmark_factory(args.benchmark)
    prompt_name = benchmark_preparation(data_obj, args.split, args)
    
    # Store preprocessed data 
    file_name = '{}_{}_{}.jsonl'.format(args.benchmark, prompt_name, args.split)
    destination_file = os.path.join(ROOT_DIR, 'benchmarks', 'ft_preprocessed', file_name)
    if args.split == 'train':
        data_obj.train_data.to_json(destination_file, orient='records', lines=True)
    elif args.split == 'validation':
        data_obj.validation_data.to_json(destination_file, orient='records', lines=True)
    elif args.split == 'test':
        data_obj.test_data.to_json(destination_file, orient='records', lines=True)
    print('Preprocessed Benchmark {} ready for fine-tuning. \nPath: {}'.format(file_name, destination_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',
                        type=str,
                        default="train",
                        help="The name of the split used for the evaluation.")
    parser.add_argument('--benchmark',
                        type=str,
                        default="medmcqa",
                        help="The name of the dataset used for the fine-tuning.")
    parser.add_argument('--cot',
                        action='store_true',
                        help="Whether to use the chain-of-thought in fine-tuning.")
    parser.add_argument('-tags', 
                        action='store_true',
                        help="Whether to use tags for the fine-tuning.")
    
    args = parser.parse_args()
    main(args)