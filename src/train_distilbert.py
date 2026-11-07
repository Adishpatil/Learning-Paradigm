"""
Train DistilBERT in TensorFlow / Keras for Learning Paradigm classification.
Can be executed locally (if TensorFlow is installed) or on Google Colab with GPU.
"""

import os
import json
import numpy as np
import pandas as pd

LABELS = ["supervised", "unsupervised", "reinforcement"]
l2i = {l: i for i, l in enumerate(LABELS)}
i2l = {i: l for i, l in enumerate(LABELS)}


def train(
    synthetic_path="data/synthetic.jsonl",
    handlabelled_path="data/handlabelled_20.csv",
    model_output_dir="models/distilbert_paradigm",
    epochs=4,
    batch_size=16,
    learning_rate=2e-5,
):
    print("Loading data...")
    syn = pd.read_json(synthetic_path, lines=True)
    hand = pd.read_csv(handlabelled_path)

    print(f"Loaded {len(syn)} synthetic scenarios and {len(hand)} hand-labelled test scenarios.")

    try:
        import tensorflow as tf
        from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
    except ImportError as e:
        print("\n[!] TensorFlow or Transformers not found in current environment.")
        print("Note: TensorFlow DistilBERT training is recommended to run on Google Colab with GPU.")
        print("See train_colab_notebook.ipynb or requirements.txt.")
        return

    print("Loading distilbert-base-uncased tokenizer...")
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def encode(texts):
        return dict(tok(list(texts), truncation=True, padding=True, max_length=128, return_tensors="tf"))

    print("Building TFDistilBertForSequenceClassification model...")
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    y_train = syn.label.map(l2i).values
    print("Starting Keras training...")
    model.fit(
        encode(syn.text),
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
    )

    print("Evaluating on hand-labelled 20 scenarios...")
    logits = model.predict(encode(hand.text)).logits
    hand["pred_bert"] = [i2l[i] for i in logits.argmax(axis=1)]
    probs = tf.nn.softmax(logits, axis=1).numpy()
    hand["bert_conf"] = probs.max(axis=1)

    acc = (hand.pred_bert == hand.label).mean()
    print(f"DistilBERT accuracy on hand-labelled set: {acc * 100:.2f}%")

    os.makedirs(model_output_dir, exist_ok=True)
    model.save_pretrained(model_output_dir)
    tok.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")

    # Save predictions
    output_eval = "data/handlabelled_20_eval.csv"
    hand.to_csv(output_eval, index=False)
    print(f"Updated predictions saved to {output_eval}")


if __name__ == "__main__":
    train()

