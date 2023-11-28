from argparse import ArgumentParser
from pathlib import Path

import datasets
from dataset import Llama2Dataset
from downsample import downsample

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--cache-dir", 
        type=Path, 
        default="/path/to/huggingface_cache/datasets",
        help="Path to huggingface cache directory")
    parser.add_argument(
        "--keep", 
        default=1.0,
        help="fraction of data to keep ")
    parser.add_argument(
        "--out", 
        type=Path, 
        default="/path/to/data/replay.json",
        help="Path of the json file to save the output")
    args = parser.parse_args()

    datasets.logging.disable_progress_bar()
    datasets.logging.set_verbosity_error()
    downsample(Llama2Dataset(cache_dir=args.cache_dir), args.keep, args.out)
