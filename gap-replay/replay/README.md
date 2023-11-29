## Replay and Code data

This directory contains the code necessary to download Replay data from RedPajama and code data from StarCoder, approximate their token count and downsample the data to a desired size. After our experiments, we decided to use 400M tokens from Replay but no code data.

Before using this code, run `huggingface-cli login` then give the key as input.
You have to accept starcoderdata TOS before getting access: https://huggingface.co/datasets/bigcode/starcoderdata

To download replay data, run:
```
python replay.py --keep=0.0004 --out=../data/replay.jsonl
```

To download code data, run: 
```
python starcoder.py --keep=0.1 --out=../data/starcoder.jsonl
```

## Token estimation

Use `count.py`.
Results:

- Estimated tokens in jupyter data: `5.110B`, computed using 80% of the data.
- Estimated tokens in starcoder without git data (including jupyter): `460.660B`, computed using 1% of the data.
- Estimated tokens in llama2: `1.054T`, computed using 3$ of the data.

**Note**: This process is relatively slow.

## Downsampling

Based on the previous results, we see that using approximately 9% of the starcoder data, we would obtain a total of around 41B tokens, which is the about the right amount for our 50% code run.
Therefore, to get that data we use:
```
python starcoder.py --keep=0.09 --out=/pure-mlo-scratch/alhernan/data/starcoder/starcoder_41B.json
```

To downsample the `experience_replay` data, use `replay.py`.
We estimate around 1T tokens in the dataset (using the 460B estimation from starcoder + official estimations from the falcon web data and redpajama data).
To get around 400M tokens, we run:
```
python replay.py --keep=0.0004 --out=/pure-mlo-scratch/alhernan/data/replay/replay-400M.json
```

- If you wish, you can also set a seed value for determinstic generation using the `--seed` argument.
- If you wish to avoid downloading the entire Falcon, RedPajama and Starcoder dataset you can use the `--streaming` flag.
  Note that this will slow down the dataset generation but will avoid storing any unneeded data.
