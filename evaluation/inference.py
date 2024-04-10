import json
import logging
import argparse
import numpy as np
import pandas as pd
import vllm
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer

from benchmarks import benchmark_factory, load_instruction

logger = logging.getLogger("meditron.evaluation.inference")
logger.setLevel(logging.INFO)

INSTRUCTIONS = {
    'truthfulqa': {'task': 'mcq', 'partition': 'validation', 'instructions': 'truthfulqa', 'cot_col': 'exp'},
    'medmcqa': {'task': 'mcq', 'partition': 'validation', 'instructions': 'medmcqa', 'cot_col': 'exp'},
    'pubmedqa': {'task': 'mcq', 'partition': 'test', 'instructions': 'pubmedqa', 'cot_col': 'long_answer'},
    'medqa': {'task': 'mcq', 'partition': 'test', 'instructions': 'medqa'},
    'medqa4': {'task': 'mcq', 'partition': 'test', 'instructions': 'medqa'},
    'medicationqa': {'task': 'open', 'partition': 'test', 'instructions': 'open_question'},
    'mmlu_medical': {'task': 'mcq', 'partition': 'test', 'instructions': 'medmcqa'},
    'mmlu_general': {'task': 'mcq', 'partition': 'test', 'instructions': 'medmcqa'},
    "blurb": {'task': 'open', 'partition': 'test', 'instructions': 'open_question'},
}

INSTRUCTIONS_SIMPLE = {
    'truthfulqa': {'task': 'mcq', 'partition': 'validation', 'instructions': 'mcp', 'cot_col': 'exp'},
    'medmcqa': {'task': 'mcq', 'partition': 'validation', 'instructions': 'mcp', 'cot_col': 'exp'},
    'pubmedqa': {'task': 'mcq', 'partition': 'test', 'instructions': 'open_question', 'cot_col': 'long_answer'},
    'medqa': {'task': 'mcq', 'partition': 'test', 'instructions': 'mcp'},
    'medqa4': {'task': 'mcq', 'partition': 'test', 'instructions': 'mcp'},
    'medicationqa': {'task': 'open', 'partition': 'test', 'instructions': 'open_question'},
    'mmlu_medical': {'task': 'mcq', 'partition': 'test', 'instructions': 'mcp'},
    'mmlu_general': {'task': 'mcq', 'partition': 'test', 'instructions': 'mcp'},
    "blurb": {'task': 'open', 'partition': 'test', 'instructions': 'open_question'},
}

def tokenizer_param(tokenizer, target, shots=0, cot=False, task_type="mcq"):
    """
    Determines the maximum number of tokens to generate for a given prompt and target.
    Also determines the stop sequence to use for generation.

    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param target: str, the target to generate
    :param shots: int, the number of shots to use for few-shot learning
    :param cot: bool, whether to use chain-or-thought or not
    :param task_type: str, the type of answer to generate (mcq or open)
    """
    max_new_tokens = len(tokenizer(target, add_special_tokens=True)['input_ids'])
    stop_seq = ["###"]
    if tokenizer.eos_token is not None:
        stop_seq.append(tokenizer.eos_token)
    if tokenizer.pad_token is not None:
        stop_seq.append(tokenizer.pad_token)

    if not cot and task_type == "mcq":
        max_new_tokens = len(tokenizer(target[0], add_special_tokens=False)['input_ids'])
        if shots > 0:
            max_new_tokens += 8
    if cot:
        max_new_tokens = 1024

    return max_new_tokens, stop_seq


def vllm_infer(client, tokenizer, prompt, stop_seq, max_new_tokens=1024, cot=False, temperature=0.0):
    """
    Generates a single output for a given input prompt using the VLLM backend (offline mode).
    Returns the output text.

    Reference:

    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param prompt: str, the prompt to generate from
    :param stop_seq: list, the stop sequence to use for generation
    :param max_new_tokens: int, the maximum number of tokens to generate
    :param cot: bool, whether to use chain-or-thought or not
    :param temperature: float, the temperature to use for sampling
    """

    response = client.generate(prompt, sampling_params=vllm.SamplingParams(
        # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=1.0,
        top_k=-1,
        top_p=1.0,
        temperature=temperature,
        stop=stop_seq,
        use_beam_search=False,
        max_tokens=max_new_tokens,
        logprobs=5
    ))

    def top_answer(logprob):
        top_token = max(logprob, key=logprob.get)
        output_text = tokenizer.decode(top_token, skip_special_tokens=True)
        return output_text

    if len(response) > 0:
        return [r.outputs[0].text for r in response]

    if not cot:
        return top_answer(response[0].outputs[0].logprobs[0])
    else:
        return response[0].outputs[0].text


def format_prompt(prompt, args):
    if args.shots > 0:
        prompt = prompt[:-1]
    if "orca" in args.checkpoint_name:
        system_msg = "You are an AI assistant who helps people find information."
        return f"<|im_start|> system\n{system_msg}<|im_end|>\n<|im_start|> question\n{prompt}<|im_end|>\n<|im_start|> answer\n"
    elif "medical" in args.checkpoint_name:
        system_msg = "You are a helpful, respectful and honest assistant." + \
        "Always answer as helpfully as possible, while being safe." + \
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content." + \
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n" + \
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct." + \
        "If you don't know the answer to a question, please don't share false information."""
        return f"<|im_start|> system\n{system_msg}<|im_end|>\n <|im_start|> user\n{prompt}<|im_end|>\n <|im_start|> assistant\n"
    elif np.any([x in args.checkpoint_name for x in ["medmcqa", "medqa", "pubmedqa"]]):
        return f"<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n"
    elif "med42" in args.checkpoint_name:
        if "Question:" in prompt:
            question = prompt.split("Question:")[1].strip()
        else:
            question = prompt
        return f'''\n<|system|>: You are a helpful medical assistant created by M42 Health in the UAE.\n<|prompter|>:{question}\n<|assistant|>:'''
    else:
        return prompt


def benchmark_infer(args, tokenizer, data, client=None, seed=1234):
    """
    Runs inference on a benchmark and stores generations in a pd.DataFrame.

    :param args: argparse.Namespace, the arguments to run the inference pipeline
    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param data: HuggingFace Dataset, the dataset to run inference on
    :param client: the client to use for querying the vLLM backend, Defaults to None
    :param seed: int, the seed to use for few-shot learning, Defaults to 1234
    return: pd.DataFrame, a DataFrame containing the scores for each answer
    """
    columns_to_save = ['prompt', 'gold']
    if 'subset' in data.features:
        columns_to_save.append('subset')
    if 'question' in data.features:
        columns_to_save.append('question')
    predictions = pd.DataFrame(data, columns=data.features)[columns_to_save]
    predictions = predictions.assign(output="Null")
    if args.multi_seed:
        predictions = predictions.assign(seed=seed)
    temperature = 0.8 if args.sc_cot else 0.0

    assert client is not None, "Client must be provided for offline inference."

    inference_data = json.loads(predictions.to_json(orient='records'))
    data_loader = DataLoader(inference_data, batch_size=args.batch_size, shuffle=False)

    batch_counter = 0
    for batch in tqdm(data_loader, total=len(data_loader), position=0, leave=True):
        prompts = [format_prompt(prompt, args) for prompt in batch["prompt"]]
        if batch_counter == 0:
            print(prompts[0])

        max_len, stop_seq = tokenizer_param(
            tokenizer, batch['gold'],
            shots=args.shots,
            cot=args.cot,
            task_type=args.task_type)
        outputs = vllm_infer(
            client, tokenizer,
            prompts, stop_seq, max_len,
            cot=args.cot, temperature=temperature)
        for prompt, out in zip(batch["prompt"], outputs):
            predictions.loc[predictions['prompt'] == prompt, 'output'] = out
        batch_counter += 1

    return predictions

def benchmark_preparation(data_obj, partition, args, seed=1234):
    """
    Runs the benchmark preparation pipeline on a given benchmark.

    :param data_obj: benchmark.Benchmark, the benchmark to run the preparation pipeline on
    :param partition: str, the partition to run the preparation pipeline on
    :param args: argparse.Namespace, the arguments to run the preparation pipeline
    """
    data_obj.load_data(partition=partition, local_path=args.local_path)
    data_obj.preprocessing(partition=partition)

    instructions = INSTRUCTIONS_SIMPLE if args.instruction == "simple" else INSTRUCTIONS
    prompt_name = instructions[args.benchmark]['instructions']

    if args.cot:
        prompt_name = prompt_name + '_cot'
    logging.info(f'Prompt used for evaluation: {prompt_name}')

    instruction = load_instruction(prompt_name)
    logging.info(f'Instruction used for evaluation: \n\t{instruction["system"]}\n\t{instruction["user"]}\n')

    if args.shots > 0:
        if not args.cot:
            logging.info('Load train data for few shot learning')
            data_obj.load_data(partition='train', local_path=args.local_path)
            data_obj.preprocessing(partition='train')

        logging.info(f'FEW SHOTS: {args.shots}')
        data_obj.add_few_shot(
            shots=args.shots,
            seed=seed,
            load_cot=args.cot)

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
    Runs the inference pipeline on a given checkpoint name and benchmark.

    :param args: argparse.Namespace, the arguments to run the inference pipeline
    """
    partition = INSTRUCTIONS[args.benchmark]['partition']
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    logging.info(f'Loaded tokenizer \n\tfrom checkpoint: {args.checkpoint}')

    data_obj = benchmark_factory(args.benchmark)
    benchmark_preparation(data_obj, partition, args)
    args.task_type = INSTRUCTIONS[args.benchmark]['task']

    kwargs = {
        "model": args.checkpoint,
        "tokenizer": args.checkpoint,
        "trust_remote_code": True,
        "max_num_seqs": 1024,
        "tensor_parallel_size": torch.cuda.device_count(),
    }

    if any([x in args.checkpoint_name for x in ["med42", "clinical-camel", "mistral", "mpt",
                                                "mistral-raw", "falcon", "zephyr"]]):
        logging.info(f"/pure-mlo-scratch/trial-runs/{args.checkpoint_name}")
        kwargs["download_dir"] = f"/pure-mlo-scratch/trial-runs/{args.checkpoint_name}"

    if "7b" in args.checkpoint:
        kwargs["tensor_parallel_size"] = 4

    client = vllm.LLM(**kwargs)

    logging.info(f'Running inference on {args.benchmark} for {len(data_obj.test_data)} samples')
    if args.shots > 0 and args.multi_seed:
        predictions = pd.DataFrame()
        for seed in [1234, 432, 32]:
            logging.info(f'Start seed {seed})')
            benchmark_preparation(data_obj, partition, args, seed=seed)
            seed_predictions = benchmark_infer(
                args, tokenizer,
                data_obj.test_data,
                client, seed=seed)
            predictions = pd.concat([predictions, seed_predictions])
            logging.info(f'Finished seed {seed}, {len(predictions)} generations collected.')
    elif args.sc_cot:
        predictions = pd.DataFrame()
        for i in range(args.sc_branch):
            logging.info(f'Start branch {i+1}/{args.sc_branch}')
            branches = benchmark_infer(args, tokenizer, data_obj.test_data, client)
            predictions = pd.concat([predictions, branches])
            logging.info(f'Finished branch {i+1}/{args.sc_branch}, {len(predictions)} generations collected.')
    else:
        predictions = benchmark_infer(args, tokenizer, data_obj.test_data, client)

    if args.cot and args.checkpoint_name in ["med42", "clinical-camel", "mistral", "mpt", "falcon", "zephyr"]:
        args.checkpoint_name = "cot" + args.checkpoint_name
    if args.sc_cot:
        args.checkpoint_name = args.checkpoint_name.replace("cot", "sc-cot")
        args.checkpoint_name = args.checkpoint_name.replace("medical", "sc-medical")
    data_obj.add_generations(data=predictions)
    data_obj.save_generations(checkpoint_name=args.checkpoint_name, shots=args.shots)
    logging.info(f'{len(predictions)} generations store for checkpoint: {args.checkpoint_name}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        type=str,
                        help="Path to the checkpoint to run inference on")
    parser.add_argument('--checkpoint_name',
                        type=str,
                        help="Name of the checkpoint to run inference on")
    parser.add_argument('--benchmark',
                        type=str,
                        default="medmcqa",
                        help="Name of the benchmark to run inference on")
    parser.add_argument('--instruction',
                        type=str,
                        default="curated",
                        help="Name of the instruction to run inference on")
    parser.add_argument('--shots',
                        type=int,
                        default=5,
                        help="Number of shots for few shot learning")
    parser.add_argument('--local_path',
                        type=str,
                        default=None,
                        help="Optional local path to the benchmark data. \
                            Defaults to None (use HuggingFace datasets).")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for few-shot evaluation.")
    parser.add_argument('--multi_seed',
                        action='store_true',
                        help="Whether to run inference on multiple seeds or not.")
    parser.add_argument('--cot',
                        action='store_true',
                        help="Whether to use chain-or-thought or not")
    parser.add_argument('--sc_cot',
                        action='store_true',
                        help="Whether to use self-consistency chain-or-thought or not")
    parser.add_argument('--sc_branch',
                        type=int,
                        default=10,
                        help="Number of branches for self-consistency chain-or-thought")
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help="Batch size for inference")
    args = parser.parse_args()
    main(args)
