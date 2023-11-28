from pathlib import Path
from argparse import ArgumentParser

import datasets

from dataset import StarcoderDataset, Llama2Dataset


def main(name: str, cache_dir: Path, vocab_file: Path): 
    if name == "starcoder":
        dset = StarcoderDataset(ignore_git=True, cache_dir=cache_dir)
    elif name == "starcoder-jupyter":
        dset = StarcoderDataset(jupyter_only=True, cache_dir=cache_dir)
    elif name == "llama2":
        dset = Llama2Dataset(cache_dir=cache_dir)
    else:
        raise KeyError(f"Unknown dataset {name}")

    dset.estimate_tokens(vocab_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument(
        "--dataset", 
        default="starcoder",
        choices=["starcoder", "starcoder-jupyter", "llama2"],
        help="Dataset to estimate tokens.")
    
    parser.add_argument(
        "--cache-dir", 
        type=Path, 
        default="/path/to/huggingface_cache/datasets",
        help="Path to huggingface cache directory.")
    
    parser.add_argument(
        "--vocab-file", 
        type=Path, 
        default="/path/to/llama/tokenizer.model",
        help="Path to the tokenizer model.")
    
    args = parser.parse_args()
    datasets.logging.disable_progress_bar()
    datasets.logging.set_verbosity_error()
    main(args.dataset, args.cache_dir, args.vocab_file)
