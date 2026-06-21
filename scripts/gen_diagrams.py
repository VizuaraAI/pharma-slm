# Technical-schematic generation via the parallel Gemini pipeline.
# Unlike gen_figures.py (decorative, NO text), these prompts demand precise LABELED diagrams.
import os, time, concurrent.futures
from pathlib import Path
from google import genai
from google.genai import types

KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBdnNPTrPDVYo1SJeOVR92_wX3HNhi1_ck")
OUT = Path(__file__).resolve().parent.parent / "book" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = ("A clean, precise, professional TECHNICAL ARCHITECTURE DIAGRAM in the style of a beautiful modern "
         "deep-learning textbook or a polished research-paper figure. Flat crisp vector look on a pure white "
         "background. Clearly LABELED rounded-rectangle boxes connected by clean directional arrows showing "
         "data flow. Refined palette: emerald-teal (#0BB47A) accents, deep charcoal-ink boxes and text, light "
         "grey connectors, generous white space. Labels in a clean modern sans-serif font, perfectly legible "
         "and correctly spelled. Uncluttered, elegant, accurate, schematic. The text labels MUST be exactly "
         "as specified and spelled correctly.")

JOBS = [
    ("diag_architecture.png", "TALL VERTICAL PORTRAIT diagram of a decoder-only Transformer language model, "
     "data flowing bottom to top. From bottom to top, stacked labeled boxes connected by upward arrows: "
     "(1) box 'Token IDs'; (2) box 'Token Embedding  (vocab 32,000 x dim 2,048)'; (3) a large tall container "
     "labeled at its top 'x 24 Decoder Layers' that contains three small stacked sub-boxes 'RMSNorm', "
     "'Self-Attention (16 heads, RoPE)', 'SwiGLU MLP'; (4) box 'Final RMSNorm'; (5) box 'Linear LM Head "
     "(weight-tied to embedding)'; (6) box 'Output logits  (32,000)'. Clean vertical schematic, white background."),

    ("diag_block.png", "WIDE diagram of ONE Transformer decoder block, the modern pre-norm design, flowing left "
     "to right. Input arrow on the left labeled 'x'. Path: 'x' -> box 'RMSNorm' -> box 'Causal Self-Attention "
     "(RoPE, 16 heads)' -> a circular plus sign (residual add) -> box 'RMSNorm' -> box 'SwiGLU Feed-Forward' -> "
     "another circular plus sign (residual add) -> output arrow 'out'. TWO curved skip-connection arrows go "
     "from before each RMSNorm directly to the matching plus sign, clearly showing the residual connections. "
     "Label the two plus signs 'add'. Clean labeled schematic, white background, teal accents."),

    ("diag_swiglu.png", "Diagram of a SwiGLU feed-forward network, left to right. Input 'x (dim 2,048)' splits "
     "into TWO parallel linear branches: top branch box 'Linear W1  (2,048 -> 5,632)' then box 'SiLU'; bottom "
     "branch box 'Linear W3  (2,048 -> 5,632)'. The two branches meet at a circular multiply sign labeled "
     "'element-wise multiply', then go into box 'Linear W2  (5,632 -> 2,048)' then output 'out (2,048)'. Clean "
     "labeled schematic, white background, teal and charcoal."),

    ("diag_attention.png", "Diagram of multi-head causal self-attention with rotary embeddings, left to right. "
     "Input 'x' fans out into three boxes 'Linear Wq', 'Linear Wk', 'Linear Wv' producing 'Q', 'K', 'V'. Q and "
     "K each pass through a box 'Apply RoPE (rotary position)'. Then Q, K, V enter a box 'Scaled Dot-Product "
     "Attention  (causal mask)'. Its output goes to box 'Concat heads' then box 'Linear Wo' then output 'out'. "
     "Annotate '16 heads, head dim 128'. Clean labeled schematic, white background."),

    ("diag_tokenizer.png", "Diagram of a byte-level BPE tokenizer turning text into token IDs, left to right. "
     "Left: a box with the text 'Pharmacokinetics of acetaminophen'. Arrow to box 'Byte-level pre-tokenize'. "
     "Arrow to box 'BPE merges  (32,000 vocab)'. Arrow to a row of small token chips reading "
     "'Pharmacokinetics | of | acetaminophen'. A small callout box says 'fertility = 1.57 tokens/word'. "
     "Below, a small comparison: 'custom tokenizer: 1-2 tokens' in teal vs 'generic tokenizer: 5-6 tokens' in "
     "grey. Clean labeled schematic, white background."),

    ("diag_datapipeline.png", "WIDE horizontal data-pipeline diagram, left to right. Left: a vertical column of "
     "labeled database-cylinder icons 'FineWeb-Edu', 'PubMed', 'PMC full-text', 'MedRAG', 'Guidelines'. Arrows "
     "converge into a box 'Shard-parallel tokenization (24 workers)'. Arrow to a row of file icons labeled "
     "'per-source .bin token files'. Arrow to a box 'Weighted mixing data loader  (~50% general / ~50% "
     "domain)'. Arrow to box 'Training batches'. Clean labeled schematic, white background, teal accents."),

    ("diag_rag.png", "WIDE horizontal Retrieval-Augmented Generation flow diagram, left to right. Box 'User "
     "question' -> box 'PubMedBERT embed' -> a cylinder 'FAISS index  (300,000 PubMed passages)' -> box "
     "'Top-k passages' -> box 'Prompt = retrieved context + question' -> box '1.3B model' -> box 'Grounded "
     "answer + cited sources'. Clean labeled schematic, white background, emerald-teal accents."),

    ("diag_pipeline6.png", "WIDE horizontal six-stage pipeline, left to right, six labeled rounded boxes joined "
     "by arrows: '1. Gather Data', '2. Train Tokenizer', '3. Define Model', '4. Pretrain', '5. Instruction-tune "
     "(SFT)', '6. Evaluate'. Each box a clean rounded rectangle with a small relevant minimalist icon above the "
     "label. Refined teal-and-charcoal palette, white background, elegant and uncluttered."),

    ("diag_experiments.png", "WIDE diagram of a parallel experiment workflow, left to right. Left cluster: seven "
     "small parallel boxes under a label '7 ablations (learning rate, mix)'. Arrow to a box 'pick best LR & "
     "mix'. Arrow to a middle cluster of four parallel boxes under a label '4-model fleet (data mix)'. Arrow to "
     "a single highlighted box 'winner'. Arrow to a larger box 'scale to 1.3B'. The winning path is in emerald "
     "teal; abandoned experiments are faded grey. Clean labeled schematic, white background."),

    ("diag_scaling_bars.png", "A clean bar chart, white background, titled area at top. Two groups of bars on an "
     "x-axis labeled 'MedMCQA' and 'PubMedQA'. In each group, a grey bar labeled '350M' and a taller emerald-"
     "teal bar labeled '1.3B'. Values shown: MedMCQA 0.29 vs 0.32; PubMedQA 0.61 vs 0.65. A dashed horizontal "
     "line near the bottom labeled 'chance 0.25'. Minimalist, legible, professional figure style."),
]


def gen(name, prompt, retries=4):
    out = OUT / name
    if out.exists() and out.stat().st_size > 9000:
        return f"skip {name}"
    full = f"{STYLE}\n\nDIAGRAM TO DRAW: {prompt}"
    for attempt in range(retries):
        try:
            client = genai.Client(api_key=KEY)
            r = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[full],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            for part in r.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image"):
                    out.write_bytes(part.inline_data.data)
                    return f"ok {name}"
        except Exception as e:
            if attempt == retries - 1:
                return f"ERR {name}: {str(e)[:90]}"
            time.sleep(2 * (attempt + 1))
    return f"FAIL {name}"


if __name__ == "__main__":
    import sys
    only = set(sys.argv[1:])
    jobs = [(n, p) for n, p in JOBS if not only or n in only or n.replace("diag_", "").replace(".png", "") in only]
    done = fails = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(gen, n, p) for n, p in jobs]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result(); done += 1
            if r.startswith(("ERR", "FAIL")):
                fails += 1
            print(f"[{done}/{len(jobs)}] {r}", flush=True)
    print(f"DONE. {done} processed, {fails} failed.")
