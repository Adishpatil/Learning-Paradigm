import os
import re
import json
import math
from typing import Dict, Optional, Tuple

LABELS = ["supervised", "unsupervised", "reinforcement"]

SYSTEM_PROMPT = """You classify ML problem scenarios as supervised, unsupervised, or reinforcement learning.
Reason step by step:
1. Does an agent take actions in an environment and learn from rewards or penalties? If yes -> reinforcement.
2. Is there a known target label for each example that the model must predict? If yes -> supervised.
3. Is the goal to find structure (clusters, groups, anomalies, reduced dimensions) with no labels? If yes -> unsupervised.

Finish your response with EXACTLY these lines:
LABEL: <supervised|unsupervised|reinforcement>
CONFIDENCE: {"supervised": 0.xx, "unsupervised": 0.xx, "reinforcement": 0.xx}
ALGORITHM: <one suitable algorithm>"""


def get_client(api_key: Optional[str] = None, provider: Optional[str] = None) -> Tuple[object, str, str]:
    """
    Returns (client, provider, model_name).
    Supports Groq (preferred when user has GROQ_API_KEY) and OpenAI.
    """
    groq_key = api_key if provider == "groq" else (api_key or os.environ.get("GROQ_API_KEY"))
    openai_key = api_key if provider == "openai" else (api_key or os.environ.get("OPENAI_API_KEY"))

    # If key is missing from environment, try loading from .streamlit/secrets.toml
    if not groq_key or not openai_key:
        for secrets_path in [
            os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml"),
            os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
        ]:
            if os.path.exists(secrets_path):
                try:
                    with open(secrets_path, "r", encoding="utf-8") as f:
                        for line in f:
                            clean_line = line.strip()
                            if clean_line.startswith("GROQ_API_KEY") and "=" in clean_line:
                                val = clean_line.split("=", 1)[1].strip().strip('"').strip("'")
                                if not groq_key and val:
                                    groq_key = val
                            elif clean_line.startswith("OPENAI_API_KEY") and "=" in clean_line:
                                val = clean_line.split("=", 1)[1].strip().strip('"').strip("'")
                                if not openai_key and val:
                                    openai_key = val
                except Exception:
                    pass

    # If explicitly groq or if groq_key is found and looks like a Groq key (starts with gsk_ or specified)
    if provider == "groq" or (groq_key and (groq_key.startswith("gsk_") or not openai_key)):
        from groq import Groq  # type: ignore
        return Groq(api_key=groq_key), "groq", "openai/gpt-oss-120b"
    elif openai_key:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=openai_key), "openai", "gpt-4o-mini"
    elif groq_key:
        from groq import Groq  # type: ignore
        return Groq(api_key=groq_key), "groq", "openai/gpt-oss-120b"
    else:
        # Fallback to Groq client instantiation which will raise clear auth error if missing
        from groq import Groq  # type: ignore
        return Groq(), "groq", "openai/gpt-oss-120b"


def parse_confidence_from_text(text: str) -> Optional[Dict[str, float]]:
    """Parse JSON or key-value confidence from CONFIDENCE: line."""
    for line in text.splitlines():
        if line.strip().upper().startswith("CONFIDENCE:"):
            raw = line.split(":", 1)[1].strip()
            # Try parsing json
            try:
                # Find JSON object in string
                match = re.search(r"\{.*?\}", raw)
                if match:
                    parsed = json.loads(match.group(0))
                    cleaned = {k.lower().strip(): float(v) for k, v in parsed.items() if k.lower().strip() in LABELS}
                    total = sum(cleaned.values())
                    if total > 0:
                        return {k: round(cleaned.get(k, 0.0) / total, 4) for k in LABELS}
            except Exception:
                pass
            
            # Try regex parsing e.g. supervised: 0.8, unsupervised: 0.1, reinforcement: 0.1
            found = {}
            for lab in LABELS:
                m = re.search(rf"{lab}\s*[:=]\s*([0-9.]+)", raw, re.IGNORECASE)
                if m:
                    found[lab] = float(m.group(1))
            total = sum(found.values())
            if total > 0:
                return {k: round(found.get(k, 0.0) / total, 4) for k in LABELS}
    return None


def label_confidence_from_logprobs(token_logprobs) -> Optional[Dict[str, float]]:
    """Read probability distribution at the first token after 'LABEL:' from OpenAI logprobs."""
    if not token_logprobs:
        return None
    tokens = [t.token for t in token_logprobs]
    seen, idx = "", None
    for i, tok in enumerate(tokens):
        seen += tok
        if "LABEL:" in seen.upper():
            idx = i + 1
            while idx < len(tokens) and not tokens[idx].strip():
                idx += 1
            break
    if idx is None or idx >= len(tokens):
        return None

    probs = {lab: 0.0 for lab in LABELS}
    if hasattr(token_logprobs[idx], "top_logprobs") and token_logprobs[idx].top_logprobs:
        for alt in token_logprobs[idx].top_logprobs:
            a = alt.token.strip().lower()
            matches = [lab for lab in LABELS if a and lab.startswith(a)]
            if len(matches) == 1:
                probs[matches[0]] += math.exp(alt.logprob)
    total = sum(probs.values())
    return {k: round(v / total, 4) for k, v in probs.items()} if total > 0 else None


def classify(scenario: str, api_key: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> dict:
    """
    Classifies an ML scenario. Works seamlessly with Groq and OpenAI.
    """
    client, detected_provider, default_model = get_client(api_key=api_key, provider=provider)
    model_name = model or default_model

    # Groq does not support logprobs=True (returns 400), whereas OpenAI does
    use_logprobs = (detected_provider == "openai")

    kwargs = {
        "model": model_name,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Scenario: {scenario}"},
        ],
    }
    if use_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""

    label = None
    algorithm = None

    for line in text.splitlines():
        up = line.strip().upper()
        if up.startswith("LABEL:"):
            candidate = line.split(":", 1)[1].strip().lower()
            candidate = re.sub(r"[^a-z]", "", candidate)
            if "unsupervised" in candidate:
                label = "unsupervised"
            elif "reinforcement" in candidate:
                label = "reinforcement"
            elif "supervised" in candidate:
                label = "supervised"
        elif up.startswith("ALGORITHM:"):
            algorithm = line.split(":", 1)[1].strip()

    # Determine confidence:
    confidence = None
    if use_logprobs and hasattr(choice, "logprobs") and choice.logprobs:
        confidence = label_confidence_from_logprobs(choice.logprobs.content)

    if not confidence:
        confidence = parse_confidence_from_text(text)

    # Cross-verify label with confidence argmax
    if confidence:
        best_label = max(confidence, key=confidence.get)
        if not label:
            label = best_label
        elif label != best_label and confidence[best_label] > 0.5:
            label = best_label

    # Fallback confidence if model did not return confidence block
    if not confidence:
        if label in LABELS:
            confidence = {l: (0.90 if l == label else 0.05) for l in LABELS}
        else:
            confidence = {l: 0.3333 for l in LABELS}

    # Ensure probabilities sum to 1.0
    s = sum(confidence.values())
    if s > 0:
        confidence = {k: round(v / s, 4) for k, v in confidence.items()}

    return {
        "label": label or "unknown",
        "algorithm": algorithm or "Standard Classifier",
        "confidence": confidence,
        "reasoning": text,
        "provider": detected_provider,
        "model": model_name,
    }


if __name__ == "__main__":
    test_scenario = "Predicting customer churn based on past subscription length, usage frequency, and previous churn records."
    print("Testing classification...")
    try:
        res = classify(test_scenario)
        print("Label:", res["label"])
        print("Algorithm:", res["algorithm"])
        print("Confidence:", res["confidence"])
        print("Provider:", res["provider"])
    except Exception as e:
        print("Error during test run (check API key):", e)

