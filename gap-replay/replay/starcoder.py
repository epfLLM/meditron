from pathlib import Path
from argparse import ArgumentParser

import datasets

from downsample import downsample
from dataset import StarcoderDataset
from utils import float_or_int


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--cache-dir", 
        type=Path, 
        default="/path/to/huggingface_cache/datasets",
        help="Path to huggingface cache directory")
    parser.add_argument(
        "--keep", 
        type=float_or_int, 
        default=1.0,
        help="Percentage of data to keep (if float), absolute total number of samples (if int)")
    parser.add_argument(
        "--out", 
        type=Path, 
        default="/path/to/data/starcoder/data.json",
        help="Path of the json file to save the output")
    args = parser.parse_args()

    datasets.logging.disable_progress_bar()
    datasets.logging.set_verbosity_error()
    priority = ["jupyter-scripts-dedup-filtered", "jupyter-structured-clean-dedup"]
    downsample(StarcoderDataset(cache_dir=args.cache_dir, ignore_git=True),
               args.keep, args.out, priority=priority)
