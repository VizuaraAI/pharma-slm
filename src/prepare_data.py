"""
Shard-parallel tokenization.

Each worker handles a slice of a source's parquet files and writes one shard:
    <data_dir>/shards/<source>/<shard_id>.{train,val}.bin   (uint16)
A finalize step concatenates the shards into <data_dir>/<source>.{train,val}.bin so the
MixDataLoader reads them unchanged. Reading parquet directly + fanning across containers
is ~100x faster than one single-stream iterator for tens of billions of tokens.
"""
from __future__ import annotations
import os, glob as _glob, argparse, time
import numpy as np
from tokenizers import Tokenizer

from sources import SOURCES, iter_texts_from_files


class BinWriter:
    def __init__(self, path, buffer_tokens=8_000_000):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "wb")
        self.buf, self.buf_len, self.total = [], 0, 0
        self.buffer_tokens = buffer_tokens

    def add(self, ids):
        self.buf.append(np.asarray(ids, dtype=np.uint16))
        self.buf_len += len(ids); self.total += len(ids)
        if self.buf_len >= self.buffer_tokens:
            self.flush()

    def flush(self):
        if self.buf:
            np.concatenate(self.buf).tofile(self.f); self.buf, self.buf_len = [], 0

    def close(self):
        self.flush(); self.f.close()


def _tokenize_iter(text_iter, source, tokenizer_path, train_path, val_path,
                   token_budget=None, val_tokens=2_000_000, batch_docs=1000):
    tok = Tokenizer.from_file(tokenizer_path)
    eot = tok.token_to_id("<|endoftext|>")
    assert eot is not None, "tokenizer missing <|endoftext|>"
    tw, vw = BinWriter(train_path), BinWriter(val_path)
    t0, total, batch = time.time(), 0, []

    def flush_batch(texts):
        nonlocal total
        for e in tok.encode_batch(texts):
            ids = e.ids + [eot]
            (vw if vw.total < val_tokens else tw).add(ids)
            total += len(ids)

    try:
        for txt in text_iter:
            batch.append(txt)
            if len(batch) >= batch_docs:
                flush_batch(batch); batch = []
                if token_budget and total >= token_budget:
                    break
        if batch and not (token_budget and total >= token_budget):
            flush_batch(batch)
    finally:
        tw.close(); vw.close()

    dt = time.time() - t0
    print(f"[prep:{source}] shard done train={tw.total/1e6:.1f}M val={vw.total/1e6:.1f}M "
          f"({total/1e6:.0f}M total, {dt:.0f}s, {total/max(dt,1e-6)/1e3:.0f}k tok/s)")
    return {"source": source, "train_tokens": tw.total, "val_tokens": vw.total, "seconds": dt}


def tokenize_files(files, source, tokenizer_path, train_path, val_path,
                   token_budget=None, val_tokens=2_000_000, batch_docs=1000):
    from sources import iter_texts_from_files
    spec = SOURCES[source]
    it = iter_texts_from_files(files, spec["text_key"])
    return _tokenize_iter(it, source, tokenizer_path, train_path, val_path,
                          token_budget, val_tokens, batch_docs)


def tokenize_stream(source, shard_index, num_shards, tokenizer_path, train_path, val_path,
                    token_budget=None, val_tokens=2_000_000, batch_docs=1000):
    from sources import iter_source_shard
    it = iter_source_shard(source, shard_index, num_shards)
    return _tokenize_iter(it, source, tokenizer_path, train_path, val_path,
                          token_budget, val_tokens, batch_docs)


def finalize_source(source, data_dir):
    """Concatenate all shard bins into <data_dir>/<source>.{train,val}.bin."""
    shard_dir = os.path.join(data_dir, "shards", source)
    out = {}
    for split in ("train", "val"):
        shards = sorted(_glob.glob(os.path.join(shard_dir, f"*.{split}.bin")))
        dest = os.path.join(data_dir, f"{source}.{split}.bin")
        total = 0
        with open(dest, "wb") as fout:
            for sp in shards:
                with open(sp, "rb") as fin:
                    while True:
                        chunk = fin.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk); total += len(chunk)
        out[split] = total // 2  # uint16 -> 2 bytes/token
        print(f"[finalize:{source}] {split}: {len(shards)} shards -> {out[split]/1e9:.3f}B tokens")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--tokenizer_path", default="/vol/tokenizer/tokenizer.json")
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--val_path", required=True)
    ap.add_argument("--token_budget", type=int, default=None)
    args = ap.parse_args()
    tokenize_files(args.files, args.source, args.tokenizer_path,
                   args.train_path, args.val_path, args.token_budget)
