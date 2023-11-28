import random
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional

import torch

from tqdm.auto import tqdm
from torch.utils.data import IterableDataset
from sentencepiece import SentencePieceProcessor
import datasets

# from utils import format_number


def format_number(x):
    return int(x)



def load(name: str, data_dir: str, explicit_datadir: bool = False, **kwargs):
    if explicit_datadir:
        return datasets.load_dataset(name, data_dir=data_dir, **kwargs)
    return datasets.load_dataset(name, data_dir, **kwargs)


def loads(name: str, names: list[str], desc: str = "",
          explicit_datadir: bool = False, **kwargs) -> list[datasets.Dataset]:

    with Pool(processes=8) as pool:
        func = partial(load, name, explicit_datadir=explicit_datadir, **kwargs)
        it = tqdm(pool.imap(func, names), desc=desc, total=len(names))
        result = list(it)
    return result


class DownsampledDataset(IterableDataset):
    def __init__(self, source: IterableDataset, indices: set[int],
                 update_fn: Callable[[dict], dict] = lambda x: x):
        self.source = source
        self.indices = indices
        self.update = update_fn
        assert len(self.indices) <= len(self.source)
        if len(indices) > 0:
            assert 0 <= min(self.indices)
            assert max(self.indices) < len(source)

    def __iter__(self):
        if len(self.indices) == 0:
            return
        for i, elem in enumerate(self.source):
            if i in self.indices:
                yield self.update(elem)

    def __len__(self) -> int:
        return len(self.indices)


class Dataset(IterableDataset):
    def __init__(self, source: IterableDataset,
                 update_fn: Callable[[dict], dict] = lambda x: x):
        self.source = source
        self.update = update_fn

    def downsample(self, keep_frac: float = 1.0) -> DownsampledDataset:
        assert 0.0 < keep_frac <= 1.0
        keep = int(len(self) * keep_frac)
        indices = list(range(len(self)))
        indices = set(random.sample(indices, keep))
        return DownsampledDataset(self.source, indices, self.update)

    def iter_rand(self):
        yield from self

    def __iter__(self):
        yield from map(self.update, self.source)

    def __len__(self) -> int:
        return len(self.source)


class DownsampledCollection(DownsampledDataset):
    def __init__(self, sources: dict[str, Dataset], keeps: dict[str, int]):
        assert set(sources) == set(keeps)
        self.sources = {name: sources[name].downsample(keep)
                        for name, keep in keeps.items()}

    def __iter__(self):
        for dset in self.sources.values():
            yield from dset

    def __len__(self) -> int:
        return sum(map(len, self.sources.values()))


def tokenize(tokenizer: SentencePieceProcessor, document: dict) -> list[int]:
    return tokenizer.encode_as_ids(document["text"])


class Collection(Dataset):
    def __init__(self, sources: dict[str, Dataset]):
        self.sources = sources

    def iter_rand(self):
        iters = {name: dset.iter_rand() for name, dset in self.sources.items()}
        weights = {name: len(dset)/len(self) for name, dset in self.sources.items()}
        read_documents = 0
        total_documents = len(self)
        while read_documents < total_documents:
            chosen_document = None
            while chosen_document is None:
                this_weights = [weights[name] for name in iters]
                it_idx, = random.choices(list(iters), weights=this_weights)
                chosen_iterator = iters[it_idx]
                try:
                    chosen_document = next(chosen_iterator)
                except StopIteration:
                    del iters[it_idx]
            yield chosen_document

    def estimate_tokens(self, vocab_file: Path, verbose: bool = True) -> int:
        token_count = 0
        char_count = 0
        read_documents = 0
        total_documents = len(self)
        tokenizer = SentencePieceProcessor(model_file=str(vocab_file))
        with Pool(processes=128) as pool:
            it = pool.imap(partial(tokenize, tokenizer), self.iter_rand())
            try:
                if verbose:
                    pbar = tqdm(desc="Estimating token count", total=total_documents)
                for tokens in it:
                    # update vars
                    token_count += len(tokens)
                    char_count += sum(map(len, tokens))
                    read_documents += 1
                    avg_tokens_per_document = token_count/read_documents
                    expected_total_tokens = avg_tokens_per_document*total_documents
                    # report progress
                    if verbose:
                        pbar.update()
                        pbar.set_postfix(
                            avg_tokens_per_document=format_number(avg_tokens_per_document),
                            expected_total_tokens=format_number(expected_total_tokens)
                        )
            except KeyboardInterrupt:
                print("Token estimation interrupted by user!")
                if verbose:
                    pbar.close()

        if verbose:
            expected_total_char = total_documents*char_count/read_documents
            print("Document count:", read_documents, "i.e.", format_number(read_documents))
            print("Total number of tokens read:", token_count, "i.e.", format_number(token_count))
            print("Estimated total number of tokens:", expected_total_tokens, "i.e.", format_number(expected_total_tokens))
            print("Estimated total number of characters:", expected_total_char, "i.e.", format_number(expected_total_char))
        return expected_total_tokens

    def downsample(self, keep_frac: float = 1.0, verbose: bool = True,
                   priority: Optional[list[str]] = None) -> DownsampledCollection:
        assert 0.0 < keep_frac <= 1.0
        keep = int(len(self) * keep_frac)
        if priority is None:
            priority = []
        priority = set(priority)

        # determine priority sizes
        keeps = {}
        total_size = sum(len(self.sources[name]) for name in priority)
        if total_size > 0:
            keep_ratio = min(1.0, keep/total_size)
            for name in priority:
                keeps[name] = int(keep_ratio*len(self.sources[name]))
            keep -= sum(keeps.values())

        # determine the sizes of the rest
        missing = set(self.sources) - priority
        total_size = sum(len(self.sources[name]) for name in missing)
        if total_size > 0:
            keep_ratio = min(1.0, keep/total_size)
            for name in missing:
                keeps[name] = int(keep_ratio*len(self.sources[name]))

        # print, if necessary
        if verbose:
            print("Number of documents per source:")
            for name, n_keep in keeps.items():
                size = len(self.sources[name])
                print(name, n_keep, "out of", size, "i.e.",
                      format_number(n_keep), "out of", format_number(size))

        return DownsampledCollection(self.sources, keeps)

    def __iter__(self):
        for dset in self.sources.values():
            yield from dset

    def __len__(self) -> int:
        return sum(map(len, self.sources.values()))


def _starcoder_update(dset_name: str, sample: dict) -> dict:
    sample["text"] = sample.pop("content")
    sample["source"] = "starcoder"
    sample["starcoder-lang"] = dset_name
    return sample


class StarcoderDataset(Collection):
    def __init__(self, ignore_git: bool = False, jupyter_only: bool = False,
                 cache_dir: Optional[Path] = None):
        # get langlist
        with open("langs.txt") as f:
            langs = list(map(lambda line: line.strip(), f))
        if ignore_git:
            langs = list(filter(lambda lang: "git" not in lang, langs))
        if jupyter_only:
            langs = list(filter(lambda lang: "jupyter" in lang, langs))

        # init loaders
        dsets = loads("bigcode/starcoderdata", langs, cache_dir=cache_dir,
                      explicit_datadir=True, desc="Getting starcoder loaders",
                      split="train")
        sources = dict(zip(langs, dsets))
        sources = {lang: Dataset(dset, update_fn=partial(_starcoder_update, lang))
                   for lang, dset in sources.items()}
        super().__init__(sources)


def _pajama_update(part_name: str, sample: dict) -> dict:
    sample["source"] = "redpajama"
    sample["pajama-block"] = part_name
    return sample


class RedPajamaDataset(Collection):
    def __init__(self, llama2_subset: bool = True,
                 cache_dir: Optional[Path] = None):
        # get names
        names = ["wikipedia", "arxiv", "book", "stackexchange"]
        if not llama2_subset:
            names += ["c4", "common_crawl", "github"]

        # init loaders
        dsets = loads("togethercomputer/RedPajama-Data-1T", names, split="train",
                      desc="Getting pajama loaders", cache_dir=cache_dir)
        sources = dict(zip(names, dsets))
        sources = {name: Dataset(dset, update_fn=partial(_pajama_update, name))
                   for name, dset in sources.items()}
        super().__init__(sources)


def _falcon_update(sample: dict) -> dict:
    sample["text"] = sample.pop("content")
    sample["source"] = "falcon-web"
    sample["timestamp"] = str(sample["timestamp"])
    return sample


class FalconDataset(Dataset):
    def __init__(self, cache_dir: Optional[Path] = None):
        print("Getting Falcon refined-web dataset")
        super().__init__(
            datasets.load_dataset("tiiuae/falcon-refinedweb", cache_dir=cache_dir,
                                  split="train"),
            update_fn=_falcon_update
        )


class Llama2Dataset(Collection):
    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__({
            # "starcoder": StarcoderDataset(cache_dir=cache_dir),
            # "falcon": FalconDataset(cache_dir=cache_dir),
            "redpajama": RedPajamaDataset(cache_dir=cache_dir)
        })
