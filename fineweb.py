"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import glob
import multiprocessing as mp
import numpy as np
import tiktoken
import pyarrow.parquet as pq
from tqdm import tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# Try to find local parquet files first (much faster than streaming)
parquet_dir = "/workspace/.hf_home/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT"
parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

if not parquet_files:
    print("No local parquet files found, falling back to streaming download...")
    from datasets import load_dataset
    fw_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
else:
    fw_stream = None
    print(f"Found {len(parquet_files)} local parquet files, reading directly (fast path)")

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize_text(text):
    """Tokenize a single text string."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count())

if parquet_files:
    print(f"Processing {len(parquet_files)} parquet files with {nprocs} workers")

shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None

def process_tokens(tokens):
    global shard_index, all_tokens_np, token_count, progress_bar
    if token_count + len(tokens) < shard_size:
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder

if parquet_files:
    with mp.Pool(nprocs) as pool:
        for pf_idx, pf_path in enumerate(parquet_files):
            print(f"Reading parquet {pf_idx+1}/{len(parquet_files)}...")
            table = pq.read_table(pf_path, columns=["text"])
            texts = [table.column("text")[i].as_py() for i in range(len(table))]
            del table
            print(f"  Tokenizing {len(texts)} documents...")
            for tokens in pool.imap(tokenize_text, texts, chunksize=256):
                process_tokens(tokens)
            del texts
else:
    # Streaming fallback (slower, for when parquets aren't available)
    def tokenize_text(text):
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(text))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
        return tokens_np.astype(np.uint16)
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize_text, (doc["text"] for doc in fw_stream), chunksize=16):
            process_tokens(tokens)

# write any remaining tokens as the last shard
if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])
