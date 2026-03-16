"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".

Optional resume mode for future interrupted runs:
$ FINEWEB_RESUME=1 python fineweb.py
"""

import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
resume_enabled = os.environ.get("FINEWEB_RESUME", "0") == "1"
worker_count = int(os.environ.get("TOKENIZE_PROCS", max(1, os.cpu_count() or 1)))

parquet_dir = "/workspace/.hf_home/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT"
parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

STATE_PATH = os.path.join(DATA_CACHE_DIR, "_resume_state.json")
BUFFER_PATH = os.path.join(DATA_CACHE_DIR, "_resume_buffer.npy")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']


def tokenize_text(text):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def tokenize_row_group(task):
    filepath, row_group_index = task
    table = pq.ParquetFile(filepath).read_row_group(row_group_index, columns=["text"])
    token_arrays = [tokenize_text(text.as_py()) for text in table.column("text")]
    if not token_arrays:
        return np.empty((0,), dtype=np.uint16)
    return np.concatenate(token_arrays)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def discover_row_group_tasks(files):
    tasks = []
    for filepath in files:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_index in range(parquet_file.num_row_groups):
            tasks.append((filepath, row_group_index))
    return tasks


def save_resume_state(task_index, shard_index, token_count, shard_buffer):
    if not resume_enabled:
        return
    with open(STATE_PATH, "w", encoding="utf-8") as state_file:
        json.dump(
            {
                "task_index": task_index,
                "shard_index": shard_index,
                "token_count": token_count,
            },
            state_file,
        )
    np.save(BUFFER_PATH, shard_buffer[:token_count])


def load_resume_state(shard_buffer):
    if not resume_enabled or not os.path.exists(STATE_PATH):
        return 0, 0, 0
    with open(STATE_PATH, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    token_count = int(state["token_count"])
    if token_count > 0 and os.path.exists(BUFFER_PATH):
        partial_tokens = np.load(BUFFER_PATH)
        shard_buffer[:token_count] = partial_tokens
    print(f"Resuming from task {state['task_index']}, shard {state['shard_index']}, buffered tokens {token_count}")
    return int(state["task_index"]), int(state["shard_index"]), token_count


def clear_resume_state():
    for path in (STATE_PATH, BUFFER_PATH):
        if os.path.exists(path):
            os.remove(path)


def process_token_block(tokens, shard_buffer, shard_index, token_count, progress_bar):
    start = 0
    tokens_len = len(tokens)
    while start < tokens_len:
        remaining = shard_size - token_count
        take = min(remaining, tokens_len - start)
        shard_buffer[token_count:token_count+take] = tokens[start:start+take]
        token_count += take
        start += take

        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(take)

        if token_count == shard_size:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, shard_buffer)
            shard_index += 1
            token_count = 0
            progress_bar = None

    return shard_index, token_count, progress_bar


def run_local_parquet_path(files):
    tasks = discover_row_group_tasks(files)
    print(f"Found {len(files)} local parquet files, {len(tasks)} row groups, using {worker_count} workers")

    shard_buffer = np.empty((shard_size,), dtype=np.uint16)
    start_task_index, shard_index, token_count = load_resume_state(shard_buffer)
    progress_bar = None
    if token_count > 0:
        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(token_count)

    pending_tasks = tasks[start_task_index:]
    with mp.Pool(worker_count) as pool:
        for completed_index, token_block in enumerate(pool.imap(tokenize_row_group, pending_tasks, chunksize=1), start=start_task_index):
            shard_index, token_count, progress_bar = process_token_block(token_block, shard_buffer, shard_index, token_count, progress_bar)
            save_resume_state(completed_index + 1, shard_index, token_count, shard_buffer)

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, shard_buffer[:token_count])
    clear_resume_state()


def run_streaming_fallback():
    print("No local parquet files found, falling back to streaming download...")
    from datasets import load_dataset

    fw_stream = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", streaming=True)
    shard_buffer = np.empty((shard_size,), dtype=np.uint16)
    shard_index = 0
    token_count = 0
    progress_bar = None

    with mp.Pool(worker_count) as pool:
        for tokens in pool.imap(tokenize_text, (doc["text"] for doc in fw_stream), chunksize=16):
            shard_index, token_count, progress_bar = process_token_block(tokens, shard_buffer, shard_index, token_count, progress_bar)

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, shard_buffer[:token_count])


if parquet_files:
    run_local_parquet_path(parquet_files)
else:
    run_streaming_fallback()
