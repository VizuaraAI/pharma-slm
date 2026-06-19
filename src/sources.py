"""
Registry of text sources + parquet-file listing for shard-parallel tokenization.

Only the two reliable, fast sources are kept (the optional PMC/pubmed25/drug-label
mirrors were broken: moved FTP path / trust_remote_code / 404). This is BioGPT's recipe
(PubMed-only domain) plus FineWeb-Edu for general English fluency.

Tokenization reads parquet files directly (fast, batched) and fans them across many
containers, instead of one slow single-stream iterator.
"""
from __future__ import annotations

SOURCES = {
    # general English (fluency). sample-10BT = ~10B tokens; we over-train so repetition is fine.
    "fineweb_edu": dict(
        repo="HuggingFaceFW/fineweb-edu", glob="sample/10BT/*.parquet",
        text_key=["text"], bucket="general", target_tokens=12_000_000_000,
    ),
    # pharma / biomedical (domain): PubMed titles+abstracts.
    "pubmed": dict(
        repo="casinca/PUBMED_title_abstracts_2019_baseline", glob="**/*.parquet",
        text_key=["text", "abstract", "article"], bucket="domain",
        target_tokens=6_000_000_000,
    ),
    # --- expanded domain corpus (added to break the 1B-token repetition wall) ---
    "medrag_pubmed": dict(   # 23.9M PubMed snippets
        repo="MedRAG/pubmed", glob="**/*.parquet",
        text_key=["content", "contents", "text"], bucket="domain",
        target_tokens=6_500_000_000,
    ),
    "pmc_commercial": dict(  # PubMed Central full-text (commercial-use subset)
        repo="TaylorAI/pubmed_commercial", glob="**/*.parquet",
        text_key=["text"], bucket="domain", target_tokens=6_000_000_000,
    ),
    "guidelines": dict(      # clinical guidelines (Meditron's GAP corpus)
        repo="epfl-llm/guidelines", glob="**/*.parquet",
        text_key=["clean_text", "text"], bucket="domain", target_tokens=600_000_000,
    ),
    "med_textbooks": dict(   # 18 medical textbooks
        repo="MedRAG/textbooks", glob="**/*.parquet",
        text_key=["content", "contents"], bucket="domain", target_tokens=40_000_000,
    ),
}


def list_parquet_files(source):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    spec = SOURCES[source]
    repo = spec["repo"]
    patterns = [
        f"datasets/{repo}/{spec['glob']}",
        f"datasets/{repo}/**/*.parquet",
        # HF auto-converts non-parquet datasets to this hidden branch:
        f"datasets/{repo}@refs/convert/parquet/**/*.parquet",
    ]
    for pat in patterns:
        try:
            paths = fs.glob(pat)
        except Exception:
            paths = []
        if paths:
            return [f"hf://{p}" for p in sorted(paths)]
    return []


def get_text(example, text_key):
    keys = text_key if isinstance(text_key, list) else [text_key]
    for k in keys:
        if k in example and example[k]:
            v = example[k]
            return v if isinstance(v, str) else " ".join(map(str, v))
    for k, v in example.items():
        if isinstance(v, str) and len(v) > 20:
            return v
    return None


def iter_texts_from_files(files, text_key, max_docs=None):
    """Stream documents from an explicit list of parquet files (one worker's shard)."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    n = 0
    for ex in ds:
        txt = get_text(ex, text_key)
        if txt and len(txt.strip()) > 1:
            yield txt
            n += 1
            if max_docs and n >= max_docs:
                return


def iter_texts(name, max_docs=None):
    """Convenience: stream a source by name (used by the tokenizer trainer / probe)."""
    spec = SOURCES[name]
    files = list_parquet_files(name)
    yield from iter_texts_from_files(files, spec["text_key"], max_docs=max_docs)


def iter_source_shard(name, shard_index, num_shards, max_docs=None):
    """Stream the FULL dataset via its native loader and take shard `shard_index` of
    `num_shards` (every num_shards-th example). Used for large sources whose HF
    auto-parquet branch is only 'partial' (MedRAG/pubmed, PMC full-text, etc.)."""
    from datasets import load_dataset
    spec = SOURCES[name]
    try:
        ds = load_dataset(spec["repo"], split="train", streaming=True)
    except Exception:
        ds = load_dataset(spec["repo"], split="train", streaming=True, trust_remote_code=True)
    ds = ds.shard(num_shards=num_shards, index=shard_index)
    n = 0
    for ex in ds:
        txt = get_text(ex, spec["text_key"])
        if txt and len(txt.strip()) > 1:
            yield txt
            n += 1
            if max_docs and n >= max_docs:
                return
