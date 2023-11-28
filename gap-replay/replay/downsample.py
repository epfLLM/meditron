import json
from dataset import Dataset
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm

def downsample(dset: Dataset, keep: float,
               out: Path, priority: Optional[list[str]] = None):

    if out.exists():
        raise FileExistsError(f"Output file {out} already exists") 

    with open(out, "w+") as f:
        for sample in tqdm(dset.downsample(keep, priority=priority), desc="Downsampling"):
            f.write(json.dumps(sample) + "\n")
