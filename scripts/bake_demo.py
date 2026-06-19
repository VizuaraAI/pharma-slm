"""Call the live endpoint for curated examples and bake real model outputs into the site."""
import json, urllib.request, os

BASE = "https://teamvizuara--pharma-slm-serve-web.modal.run"


def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


CHAT = [
    "What is the mechanism of action of metformin?",
    "What are the common side effects of warfarin?",
    "What is amoxicillin used to treat?",
    "What is the mechanism of action of aspirin?",
    "What is ibuprofen used for?",
    "What is hypertension and how is it treated?",
]

MCQ = [
    {"question": "Which of the following is a proton pump inhibitor?",
     "options": ["Omeprazole", "Atenolol", "Metformin", "Warfarin"], "answer": 0, "kind": "recall"},
    {"question": "Which drug is a beta-blocker used to treat hypertension?",
     "options": ["Atenolol", "Omeprazole", "Amoxicillin", "Insulin"], "answer": 0, "kind": "recall"},
    {"question": "What is the first-line oral medication for type 2 diabetes?",
     "options": ["Metformin", "Atorvastatin", "Lisinopril", "Omeprazole"], "answer": 0, "kind": "recall"},
    {"question": "Which antibiotic class does amoxicillin belong to?",
     "options": ["Beta-lactam (penicillin)", "Macrolide", "Fluoroquinolone", "Tetracycline"], "answer": 0, "kind": "recall"},
    {"question": "A patient on warfarin is prescribed an antibiotic. Which most increases bleeding risk?",
     "options": ["Metronidazole", "Cephalexin", "Azithromycin", "Nitrofurantoin"], "answer": 0, "kind": "reasoning"},
    {"question": "A 60-year-old on an ACE inhibitor develops a dry cough. The mechanism is accumulation of:",
     "options": ["Bradykinin", "Angiotensin II", "Histamine", "Serotonin"], "answer": 0, "kind": "reasoning"},
]


def main():
    print("baking chat...")
    chat = []
    for p in CHAT:
        a = post("/generate", {"prompt": p, "max_new_tokens": 110, "temperature": 0.6})["answer"]
        chat.append({"q": p, "a": a})
        print("  ✓", p)
    print("baking mcq...")
    mcq = []
    for m in MCQ:
        r = post("/mcq", {"question": m["question"], "options": m["options"], "answer": m["answer"]})
        mcq.append({**m, **r})
        print(f"  ✓ pred={r['pred']} correct={r['correct']}  {m['question'][:50]}")
    os.makedirs("site", exist_ok=True)
    json.dump({"chat": chat, "mcq": mcq}, open("site/demo_data.json", "w"), indent=2)
    n_right = sum(x["correct"] for x in mcq)
    print(f"wrote site/demo_data.json  | mcq right: {n_right}/{len(mcq)}")


if __name__ == "__main__":
    main()
