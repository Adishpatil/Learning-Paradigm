import os
import json
import pandas as pd
import numpy as np
try:
    from src.cot_classifier import classify, LABELS
except (ImportError, ModuleNotFoundError):
    from cot_classifier import classify, LABELS

FEW_SHOT_PROMPT_ADDITION = """
Here are reference examples:
Example 1:
Scenario: Predicting quarterly loan default rates for small business borrowers using credit bureau records and verified repayment histories.
Reasoning: There is a known historical ground-truth outcome (default vs non-default).
LABEL: supervised
ALGORITHM: Gradient Boosting / XGBoost

Example 2:
Scenario: Grouping customers based on unlabelled browsing patterns and click sequences without pre-defined segments.
Reasoning: No ground truth labels are provided; the goal is to discover latent clusters.
LABEL: unsupervised
ALGORITHM: K-Means Clustering

Example 3:
Scenario: A heating and cooling controller adjusting thermostat setpoints every 15 minutes to minimize power consumption while keeping room temperature in a comfort zone using thermal feedback.
Reasoning: An agent takes sequential actions in an environment, receiving reward signals (energy saved, temperature comfort penalty).
LABEL: reinforcement
ALGORITHM: Deep Q-Network (DQN) / PPO
"""


def evaluate_llm(data_path="data/handlabelled_20.csv", few_shot=False, api_key=None, provider=None):
    df = pd.read_csv(data_path)
    print(f"Running LLM evaluation (few_shot={few_shot}) on {len(df)} scenarios...")
    
    preds = []
    confs = []
    algs = []
    
    for idx, row in df.iterrows():
        text = row["text"]
        print(f"[{idx+1}/{len(df)}] Evaluating: {text[:50]}...")
        try:
            res = classify(text, api_key=api_key, provider=provider)
            pred_label = res["label"]
            conf = res["confidence"].get(pred_label, 0.0) if res["confidence"] else 0.0
            alg = res["algorithm"]
        except Exception as e:
            print(f"  Error on item {idx+1}: {e}")
            pred_label = "error"
            conf = 0.0
            alg = "None"
            
        preds.append(pred_label)
        confs.append(conf)
        algs.append(alg)
        
    col_pred = "pred_few_shot" if few_shot else "pred_llm"
    col_conf = "conf_few_shot" if few_shot else "conf_llm"
    col_alg = "alg_few_shot" if few_shot else "alg_llm"
    
    df[col_pred] = preds
    df[col_conf] = confs
    df[col_alg] = algs
    return df


def print_evaluation_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("           EVALUATION SUMMARY & DOMAIN BIAS ANALYSIS")
    print("=" * 60)

    pred_cols = [c for c in ["pred_llm", "pred_few_shot", "pred_bert"] if c in df.columns]

    if not pred_cols:
        print("No prediction columns found to evaluate.")
        return

    for c in pred_cols:
        df[f"ok_{c}"] = (df[c] == df["label"]).astype(int)

    # Overall Metrics
    for c in pred_cols:
        acc = df[f"ok_{c}"].mean()
        print(f"Overall Accuracy [{c}]: {acc * 100:.2f}% ({df[f'ok_{c}'].sum()}/{len(df)})")

    # Per-domain Breakdown
    print("\nPer-Domain Breakdown:")
    domain_group = df.groupby("domain")[[f"ok_{c}" for c in pred_cols]].agg(["count", "mean"])
    print(domain_group)

    # Per-label Breakdown
    print("\nPer-Paradigm Label Breakdown:")
    label_group = df.groupby("label")[[f"ok_{c}" for c in pred_cols]].agg(["count", "mean"])
    print(label_group)

    # Disagreements / Errors
    print("\nError Analysis & Edge Cases:")
    for c in pred_cols:
        errors = df[df[f"ok_{c}"] == 0]
        print(f"\nMisclassifications for [{c}] ({len(errors)} total):")
        for _, row in errors.iterrows():
            print(f" - [{row['domain']}] True: '{row['label']}' | Pred: '{row[c]}'")
            print(f"   Text: {row['text']}")


def run_full_evaluation(input_csv="data/handlabelled_20.csv", output_csv="data/handlabelled_20_eval.csv"):
    df = pd.read_csv(input_csv)
    
    # Check if predictions already exist or evaluate
    if "pred_llm" not in df.columns:
        df = evaluate_llm(input_csv, few_shot=False)
        
    print_evaluation_summary(df)
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed evaluation written to {output_csv}")


if __name__ == "__main__":
    run_full_evaluation()

