"""
Mixing data loader.

Each source is pre-tokenized into its own flat token file: <data_dir>/<source>.<split>.bin
(uint16, since our vocab is 32k < 65536). At batch time we sample which source each
sequence comes from according to `mix` weights, then take a random contiguous window.

This means the general/domain ratio is a *runtime* knob — the sweep can try many mix
ratios without ever re-tokenizing. Reads via np.memmap so nothing loads into RAM.
"""
from __future__ import annotations
import os
import numpy as np
import torch


class MixDataLoader:
    def __init__(self, data_dir, mix, seq_len, batch_size, device, split="train", seed=1337):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.split = split
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"

        sources, weights, sizes = [], [], {}
        for name, w in mix.items():
            path = os.path.join(data_dir, f"{name}.{split}.bin")
            if not os.path.exists(path):
                # val split may be missing for tiny sources; fall back to train
                alt = os.path.join(data_dir, f"{name}.train.bin")
                if os.path.exists(alt):
                    path = alt
                else:
                    print(f"[data] WARNING: missing {path}, dropping source '{name}'")
                    continue
            if os.path.getsize(path) == 0:
                print(f"[data] WARNING: empty {path}, dropping source '{name}'")
                continue
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            if len(arr) <= seq_len + 1:
                print(f"[data] WARNING: source '{name}' too small ({len(arr)} toks), dropping")
                continue
            sources.append((name, path, len(arr)))
            weights.append(float(w))
            sizes[name] = len(arr)

        if not sources:
            raise RuntimeError(f"No usable sources in {data_dir} for split={split}, mix={mix}")

        self.sources = sources
        w = np.array(weights, dtype=np.float64)
        self.weights = w / w.sum()
        self.sizes = sizes
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 99))
        total = sum(sz for _, _, sz in sources)
        self.total_tokens = total
        print(f"[data] split={split} sources=" +
              ", ".join(f"{n}:{sz/1e6:.1f}M(w={ww:.2f})"
                        for (n, _, sz), ww in zip(sources, self.weights)) +
              f"  total={total/1e9:.3f}B toks")

    def _memmap(self, path):
        # recreate each call to avoid the well-known memmap memory leak
        return np.memmap(path, dtype=np.uint16, mode="r")

    def get_batch(self):
        # choose a source per sequence in the batch
        choices = self.rng.choice(len(self.sources), size=self.batch_size, p=self.weights)
        xs, ys = [], []
        for ci in choices:
            _, path, n = self.sources[ci]
            data = self._memmap(path)
            i = int(self.rng.integers(0, n - self.seq_len - 1))
            chunk = data[i:i + self.seq_len + 1].astype(np.int64)
            xs.append(torch.from_numpy(chunk[:-1]))
            ys.append(torch.from_numpy(chunk[1:]))
        x = torch.stack(xs)
        y = torch.stack(ys)
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
