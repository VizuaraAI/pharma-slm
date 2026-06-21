# Bulk figure generation for the book — direct Gemini calls fanned across threads.
# pip install google-genai ; GEMINI_API_KEY=... python scripts/gen_figures.py
import os, time, concurrent.futures
from pathlib import Path
from google import genai
from google.genai import types

KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBdnNPTrPDVYo1SJeOVR92_wX3HNhi1_ck")
OUT = Path(__file__).resolve().parent.parent / "book" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = ("Premium minimalist conceptual illustration for a high-end technical book on building "
         "artificial-intelligence language models for medicine. Soft teal-emerald and deep "
         "charcoal-ink palette on a warm off-white background, subtle gradients, delicate glow, "
         "clean modern editorial style, elegant, generous negative space, refined and premium. "
         "NO text, no words, no numbers, no letters, no captions, no labels anywhere in the image.")

JOBS = [
    ("fig_cover.png", "A small luminous brain woven from fine neural filaments rising out of an open medical book and floating molecules and pills, radiating soft teal light; a sense of something being built from nothing."),
    ("fig_pipeline.png", "A clean horizontal assembly line of six glowing stations that progressively transform a stream of raw paper documents on the left into a single glowing crystalline orb of intelligence on the right."),
    ("fig_data.png", "Countless medical research papers, molecules and DNA strands flowing like a river into a wide funnel, distilling into a concentrated glowing droplet."),
    ("fig_mixing.png", "Two distinct rivers — one pale and broad representing everyday text, one emerald and dense representing medicine — merging in balanced proportion into a single calm confluence."),
    ("fig_data_ceiling.png", "A tall glass reservoir half-filled with glowing emerald liquid that has clearly hit a hard ceiling, a small specialized well running low beside an enormous ordinary lake."),
    ("fig_tokenizer.png", "A single very long elegant ribbon being cleanly snapped into tidy interlocking puzzle pieces of varying size by a precise mechanism, soft glow at each cut."),
    ("fig_architecture.png", "An elegant tall layered tower of translucent stacked plates connected by glowing vertical and diagonal threads of light, like a cathedral of computation."),
    ("fig_attention.png", "A row of abstract glowing orbs in a line, with luminous curved threads connecting each orb to several others, some threads brighter than others, depicting selective focus."),
    ("fig_pretraining.png", "A vast dark hall filled with neat rows of glowing server racks forming a single immense pulsing engine, beams of teal light converging at the centre."),
    ("fig_overtraining.png", "A single small seedling in rich soil being patiently watered far longer than usual, growing unusually strong roots and a bright sturdy sprout."),
    ("fig_research_team.png", "A row of identical small friendly scientist-robots each tending its own glowing experiment simultaneously, a sense of a coordinated parallel research team."),
    ("fig_ablation.png", "Many small glowing test tubes arranged in a grid; a few glow bright and triumphant while several dim and fade, depicting experiments being kept or abandoned."),
    ("fig_fleet.png", "Four parallel glowing furnaces or forges of slightly different colour-balance running side by side in a race, sparks rising, one pulling subtly ahead."),
    ("fig_overfitting.png", "On one side a figure obsessively memorising a single closed book until it glows too hot; on the other a calm figure understanding an open landscape of many books."),
    ("fig_sft.png", "A rough unformed glowing clay figure being gently shaped by unseen hands into a poised attentive assistant that is clearly listening and ready to answer."),
    ("fig_eval.png", "A glowing orb of intelligence seated at a desk taking a written examination, a report card with abstract marks materialising beside it."),
    ("fig_capacity.png", "A small glowing brain straining against an intricate oversized mechanical puzzle far too complex for it, a few gears just out of reach."),
    ("fig_scaling.png", "A small glowing orb of intelligence growing in three steps into a much larger brighter orb, gaining structure and radiance as it grows."),
    ("fig_rag.png", "A glowing mind reaching one luminous tendril into a vast towering library of medical volumes, pulling out three bright passages that anchor and ground its thought."),
    ("fig_infra.png", "A calm modern control room with a single large dashboard overseeing distant glowing clusters of cloud GPUs, serene and in command."),
    ("fig_resume.png", "A determined glowing process moving along a path through a sudden storm, briefly interrupted, then seamlessly continuing from exactly where it paused."),
    ("fig_cost.png", "An ascending staircase of progressively larger glowing model-orbs, each step taller and brighter, with rising stacks of abstract energy beneath each step."),
    ("fig_bugs.png", "Tiny mischievous gremlin-shaped glitches being carefully caught one by one inside delicate clockwork machinery and the machine then running smoothly."),
    ("fig_production.png", "A polished refined assistant-orb gently serving several people at once from behind a soft protective glass guardrail, trustworthy and calm."),
    ("fig_roadmap.png", "A winding luminous path climbing a stylised mountain through several waypoints toward a bright summit, a clear sense of a journey and a destination."),
    ("fig_future.png", "A glowing model-orb evolving toward a brighter larger more intricate future form on the horizon at sunrise, hopeful and expansive."),
    ("fig_spark.png", "A single tiny brilliant spark igniting in vast dark negative space, the very beginning of something, minimal and poetic."),
    ("fig_grounding.png", "A floating answer made of light tethered by fine luminous threads down to three solid open books beneath it, anchored to evidence."),
    ("fig_budget.png", "An elegant balance scale weighing a small stack of glowing coins against a glowing brain, the trade-off between cost and capability."),
    ("fig_enterprise.png", "A grand modern enterprise building with a glowing intelligent core at its heart, teal light flowing out to many connected smaller offices."),
]


def gen(name, prompt, retries=3):
    out = OUT / name
    if out.exists() and out.stat().st_size > 9000:
        return f"skip {name}"
    full = f"{STYLE}\n\nSCENE: {prompt}"
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
    done = fails = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as ex:
        futs = [ex.submit(gen, n, p) for n, p in JOBS]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result(); done += 1
            if r.startswith(("ERR", "FAIL")):
                fails += 1
            print(f"[{done}/{len(JOBS)}] {r}  (fails={fails})", flush=True)
    print(f"DONE. {done} processed, {fails} failed.")
