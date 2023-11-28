import random
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
        type=float,
        default=1.0,
        help="fraction of data to keep ")
    parser.add_argument(
        "--out", 
        type=Path, 
        default="/path/to/data/replay.json",
        help="Path of the json file to save the output")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Seed for reproducibility")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Don't download the entire dataset, stream from it instead (slower)")
    args = parser.parse_args()

    random.seed(args.seed)
    datasets.logging.disable_progress_bar()
    datasets.logging.set_verbosity_error()
    downsample(Llama2Dataset(cache_dir=args.cache_dir, streaming=args.streaming),
               args.keep, args.out)
