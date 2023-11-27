# Finetuning

Run `sft.py` to start the fine-tuning pipeline.
Make sure to edit the paths defined in the `CHECKPOINTS` constant in the script to suit your configuration.
Feel free to edit the `N_DOCS` dictionary as well with the number of documents in your dataset.

Example run:
```
python finetuning/sft.py --checkpoint=meditron --size=7 --run_name=pubmedqa
```

Note that if you run the fine-tuning pipeline twice with the same configuration, the script will either resume the fine-tune process starting from the latest checkpoint saved, or crash if the fine-tune has already finished before.

More information about the arguments:
- `--checkpoint`: **the name of the base model to fine-tune**.
  This should be an identifier to the pre-trained model you wish to load weights from, for instance the "baseline" llama model or the "meditron" pre trained model.
  The checkpoint path will need to be defined in the `CHECKPOINTS` constant of the `sft.py` file.
- `--size`: **the size of the base model you wish to fine-tune**.
  For example, 7 for 7B models or 70B.
  The size and the checkpoint values will identify where to load the base checkpoint from using `CHECKPOINTS[args.checkpoint, args.size]`.
- `--save_checkpoint_dir`: **the root directory to save the fine-tuned model**.
  The script will create a new directory that corresponds to the overall configuration given (e.g. base checkpoint, run_name, etc) under this root.
- `--run_name`: **the name of the dataset to use**.
  This will also be looked up in the `N_DOCS` constant directory to look for the number of documents of the dataset.
  You don't have to modify the `N_DOCS` as the script will automatically try to infer the size but will take longer to run if you don't.
- `--tokenized_data_dir`: **the root directory to load and save tokenized data**.
  The script will create a new directory under this root with the name of the dataset used (the run_name value) when saving tokenized data.
- `--data`: **paths of the jsonl files to use as training data**.
  This value should only be set the first time you run the script with a new dataset.
  For all following executions, the pre-tokenized data under tokenized_data_dir/run_name will be loaded instead, regardless of this argument.
- `--val`: **paths of the jsonl files to use as validation data**.
  Similar to the previous argument but for validation data, if needed.
  Note that for following executions, you should not set this argument empty if you want to use validation set.
  Instead, set it to any value and the script will automatically try to fetch the previously tokenized validation data.
- `--epochs`: **number of epochs to fine-tune**.
- `--no_autoaccept_iters`: **when set, you will be asked for confirmation when setting the number of iterations**.
  Normally, the iteration calculation will be set such that you run for at least the specified number of epochs, taking into account the global batch size of 64.
  If you set this flag, this infer will not be accepted automatically, but rather you will be asked if you accept the computed number of iterations.
- `--intermediate_iter`: **iteration of the base model**.
  If you wish to fine-tune an early release of the base model, you can set this value to any number.
  The script will try to load the base checkpoint (the checkpoint argument) state of this iteration.
- `--question_key`: **the key in the jsonl files that corresponds to the user's question**.
- `--answer_key`: **the key in the jsonl files that corresponds to the assistant's answer**.
- `--system_key`: **the key in the jsonl files that corresponds to the system message**.
- `--micro_batch`: **micro batch size**.
- `--seq`: **sequence length**.
- `--rank`: **rank of this process in the distributed trainer**.
- `--nodes`: **number of nodes in the distributed trainer**.
- `--addr`: **master address of the distributed trainer**.
- `--loss_mask`: **value to set as weight in the loss for the user tokens**.
  The default of zero means that the loss computed on the user tokens will not be propagated.
- `--save_interval`: **indicates how often to save fine-tuning checkpoints**.
- `--id`: **unique identifier for this experiment**.
  This can be useful if you wish to repeat fine-tuning with the same settings, but starting from scratch again.
- `--tp`: **tensor parallelism level**.
  If not set, it will be automatically inferred from the checkpoint path.
- `--pp`: **pipeline parallelism level**.
  If not set, it will be automatically inferred from the checkpoint path.
