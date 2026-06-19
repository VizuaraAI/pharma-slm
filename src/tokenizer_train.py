"""
Train a custom 32k byte-level BPE on a blend of general + pharma text.

Training the tokenizer on the *mixed* corpus (not domain-only) keeps general English
efficient while still giving single tokens to biomedical terms (e.g. 'chromatography',
'pharmacokinetics'). Special tokens for the chat/SFT format are reserved up front so we
never have to resize embeddings later.
"""
from __future__ import annotations
import os, argparse, itertools
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import ByteLevel as ByteLevelProcessor

from sources import SOURCES, iter_texts

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>"]


def blended_corpus_iter(docs_per_source, only_sources=None):
    """Round-robin a capped number of docs from each (resolvable) source."""
    iters = {}
    names = only_sources if only_sources else list(SOURCES)
    for name in names:
        try:
            iters[name] = iter_texts(name, max_docs=docs_per_source.get(name, 0))
            print(f"[tok] opened source {name}")
        except Exception as e:
            print(f"[tok] skip {name}: {e}")
    # round-robin so the sample reflects the blend, not just the first source.
    # Truncate each doc: BPE only needs local context, and bounding doc length keeps
    # the trainer's word table small/fast (un-truncated FineWeb docs blow up memory).
    active = dict(iters)
    n = 0
    while active:
        for name in list(active):
            try:
                yield next(active[name])[:5000]
                n += 1
                if n % 50000 == 0:
                    print(f"[tok] fed {n} docs", flush=True)
            except StopIteration:
                del active[name]
            except Exception as e:
                print(f"[tok] iter error {name}: {e}")
                del active[name]


def train_tokenizer(out_path, vocab_size=32000,
                    general_docs=1_200_000, domain_docs=1_200_000, only_sources=None):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = ByteLevelProcessor(trim_offsets=False)

    docs_per_source = {}
    for name, spec in SOURCES.items():
        docs_per_source[name] = general_docs if spec["bucket"] == "general" else domain_docs

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,   # progress bar spams container logs (hit log resource limits)
    )
    tok.train_from_iterator(blended_corpus_iter(docs_per_source, only_sources), trainer=trainer)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tok.save(out_path)
    print(f"[tok] saved tokenizer -> {out_path}  vocab={tok.get_vocab_size()}")

    # quick fertility sanity check on a domain string
    sample = ("Pharmacokinetics of acetaminophen: hepatic glucuronidation and "
              "cytochrome P450-mediated metabolism in patients with hepatic impairment.")
    ids = tok.encode(sample).ids
    print(f"[tok] sample tokens={len(ids)} for {len(sample.split())} words "
          f"(fertility={len(ids)/len(sample.split()):.2f})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vol/tokenizer/tokenizer.json")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--general_docs", type=int, default=1_200_000)
    ap.add_argument("--domain_docs", type=int, default=1_200_000)
    args = ap.parse_args()
    train_tokenizer(args.out, args.vocab_size, args.general_docs, args.domain_docs)
