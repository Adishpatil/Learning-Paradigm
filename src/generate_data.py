import os
import json
import random
from typing import List, Dict
try:
    from src.cot_classifier import get_client  # type: ignore
except (ImportError, ModuleNotFoundError):
    from cot_classifier import get_client  # type: ignore

LABELS = ["supervised", "unsupervised", "reinforcement"]
DOMAINS = ["healthcare", "finance", "retail", "manufacturing", "education"]


def generate(label: str, domain: str, n: int = 14, client=None, model: str = None) -> List[str]:
    """Generates n synthetic scenarios for a given label and domain."""
    prompt = (
        f"Write {n} varied, realistic ML problem scenarios (1-3 sentences each) from the "
        f"{domain} domain that clearly require {label} learning. Vary scale, wording and context. "
        'Return JSON with key "scenarios": ["...", ...]'
    )
    
    r = client.chat.completions.create(
        model=model,
        temperature=1.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    content = r.choices[0].message.content
    data = json.loads(content)
    return data.get("scenarios", [])


def generate_dataset(output_path: str = "data/synthetic.jsonl", scenarios_per_group: int = 14):
    client, provider, default_model = get_client()
    print(f"Using provider: {provider}, model: {default_model}")
    
    rows: List[Dict[str, str]] = []
    
    for lab in LABELS:
        for dom in DOMAINS:
            print(f"Generating for label='{lab}', domain='{dom}'...")
            try:
                scenarios = generate(lab, dom, n=scenarios_per_group, client=client, model=default_model)
                for s in scenarios:
                    rows.append({"text": s.strip(), "label": lab, "domain": dom})
            except Exception as e:
                print(f"  Error generating {lab}/{dom}: {e}")

    # Remove exact duplicates
    rows = list({r["text"]: r for r in rows}.values())
    random.seed(42)
    random.shuffle(rows)
    rows = rows[:200]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Successfully saved {len(rows)} scenarios to {output_path}")
    return rows


if __name__ == "__main__":
    generate_dataset()

