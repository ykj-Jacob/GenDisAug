"""
Reproduction script for:
"Generative or Discriminative? Revisiting Text Classification in the Era of Transformers"
EMNLP 2025 Outstanding Paper

BERT Encoder (discriminative) vs GPT-2 (generative) on SST-2 / AG News
Using pretrained models + fine-tuning
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset

# Add augmentation module to path
sys.path.insert(0, str(Path(__file__).parent / "augmentation"))
sys.path.insert(0, str(Path(__file__).parent))

from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = Path(__file__).parent / "results" / "reproduction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "sst2": "SetFit/sst2",
    "ag_news": "ag_news",
}


def get_balanced_subset(dataset, n_samples, seed):
    """Get balanced subset of training data."""
    if n_samples == -1:
        return {"train": dataset["train"], "test": dataset["test"] if "test" in dataset else dataset["validation"]}

    train_data = dataset["train"].shuffle(seed=seed)
    n_per_class = max(1, n_samples // len(set(train_data["label"])))

    indices = []
    labels_arr = np.array(train_data["label"])
    for label in sorted(set(labels_arr.tolist())):
        label_indices = np.where(labels_arr == label)[0]
        selected = label_indices[:n_per_class]
        indices.extend(selected.tolist())

    indices = sorted(indices)
    train_subset = train_data.select(indices)
    test_data = dataset["test"] if "test" in dataset else dataset["validation"]

    return {"train": train_subset, "test": test_data}


def run_bert_encoder(dataset_name, n_samples, seed, output_dir, augment=None):
    """BERT Encoder (discriminative fine-tuning)."""
    print(f"\n{'='*60}")
    aug_str = f" | Augment={augment}" if augment else ""
    print(f"BERT Encoder | {dataset_name} | K={n_samples} | seed={seed}{aug_str}")
    print(f"{'='*60}")

    dataset = load_dataset(DATASETS[dataset_name])
    subset = get_balanced_subset(dataset, n_samples, seed)
    num_labels = len(set(subset["train"]["label"]))

    # Apply augmentation to training data
    train_texts = subset["train"]["text"]
    train_labels = [int(l) for l in subset["train"]["label"]]

    if augment:
        from augmentation import augment_dataset
        train_texts, train_labels = augment_dataset(
            train_texts, train_labels, method=augment, alpha=0.1, num_aug=1, seed=seed
        )
        print(f"  Augmented from {len(subset['train']['text'])} → {len(train_texts)} samples")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    # Build new dataset dict with augmented train data
    from datasets import Dataset, DatasetDict
    aug_train = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_data = subset["test"]

    tokenized = {}
    tokenized["train"] = aug_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized["train"] = tokenized["train"].rename_column("label", "labels")
    tokenized["test"] = test_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized["test"] = tokenized["test"].rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        save_total_limit=1,
        report_to="none",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    predictions = trainer.predict(tokenized["test"])
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    result = {"model": "bert_encoder", "dataset": dataset_name, "n_samples": n_samples,
              "seed": seed, "augment": augment, "accuracy": float(acc), "macro_f1": float(f1)}
    print(f"Result: Acc={acc:.4f}, F1={f1:.4f}")
    return result


def run_gpt2_ar(dataset_name, n_samples, seed, output_dir, augment=None):
    """GPT-2 generative classifier — pretrained GPT-2 fine-tuned on P(label|text)."""
    print(f"\n{'='*60}")
    aug_str = f" | Augment={augment}" if augment else ""
    print(f"GPT-2 AR (Generative) | {dataset_name} | K={n_samples} | seed={seed}{aug_str}")
    print(f"{'='*60}")

    from torch.utils.data import Dataset, DataLoader
    from transformers import get_linear_schedule_with_warmup

    seed_everything(seed)

    dataset = load_dataset(DATASETS[dataset_name])
    subset = get_balanced_subset(dataset, n_samples, seed)
    num_labels = len(set(subset["train"]["label"]))
    all_lbls = list(range(num_labels))

    # Apply augmentation
    train_texts_list = list(subset["train"]["text"])
    train_labels_list = [int(l) for l in subset["train"]["label"]]

    if augment:
        from augmentation import augment_dataset
        train_texts_list, train_labels_list = augment_dataset(
            train_texts_list, train_labels_list, method=augment, alpha=0.1, num_aug=1, seed=seed
        )
        print(f"  Augmented from {len(subset['train']['text'])} → {len(train_texts_list)} samples")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.pad_token_id

    label_token_ids = [tokenizer.encode(f" {l}", add_special_tokens=False)[0] for l in all_lbls]

    # Build training data: "Text: {text}\nLabel: {label}"
    def build_prompt(text, label=None):
        prompt = f"Text: {text}\nLabel:"
        if label is not None:
            prompt += f" {label}"
        return prompt

    # Tokenize training data (use augmented data if available)
    train_encodings = tokenizer(
        [build_prompt(t, l) for t, l in zip(train_texts_list, train_labels_list)],
        padding=True, truncation=True, max_length=256, return_tensors="pt"
    )

    class GenDataset(Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.input_ids[idx].clone(),
            }

    train_ds = GenDataset(train_encodings)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_steps = min(len(train_loader) * 5, 500)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_steps)

    model.train()
    global_step = 0
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            global_step += 1
        print(f"  Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate: for each test example, score each label
    model.eval()
    val_texts = subset["test"]["text"]
    val_labels = [int(l) for l in subset["test"]["label"]]

    preds = []
    with torch.no_grad():
        for text in val_texts:
            label_scores = []
            for label in all_lbls:
                prompt = build_prompt(text, label)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs["input_ids"])
                # Negative loss = higher likelihood
                label_scores.append(-outputs.loss.item())
            preds.append(all_lbls[np.argmax(label_scores)])

    acc = accuracy_score(val_labels, preds)
    f1 = f1_score(val_labels, preds, average="macro")
    result = {"model": "gpt2_ar", "dataset": dataset_name, "n_samples": n_samples,
              "seed": seed, "augment": augment, "accuracy": float(acc), "macro_f1": float(f1)}
    print(f"Result: Acc={acc:.4f}, F1={f1:.4f}")
    return result


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "ag_news"])
    parser.add_argument("--models", type=str, default="bert_encoder,gpt2_ar")
    parser.add_argument("--sample_sizes", type=str, default="128,512,2048,-1")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--augment", type=str, default=None,
                       choices=["eda", "backtrans", "both"],
                       help="Data augmentation method")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    sample_sizes_use = [128] if args.quick else [int(s) for s in args.sample_sizes.split(",")]
    seeds_use = [42] if args.quick else [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")

    # Set results directory based on augmentation
    if args.augment:
        base_dir = RESULTS_DIR.parent / f"augmentation_{args.augment}"
    else:
        base_dir = RESULTS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = base_dir / f"results_{timestamp}.json"

    for model_type in models:
        for n_samples in sample_sizes_use:
            for seed in seeds_use:
                aug_tag = f"_{args.augment}" if args.augment else ""
                exp_name = f"{model_type}_{args.dataset}_K{n_samples}_s{seed}{aug_tag}"
                output_dir = base_dir / exp_name
                output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    if model_type == "bert_encoder":
                        result = run_bert_encoder(args.dataset, n_samples, seed, output_dir, args.augment)
                    elif model_type == "gpt2_ar":
                        result = run_gpt2_ar(args.dataset, n_samples, seed, output_dir, args.augment)
                    else:
                        continue

                    all_results.append(result)
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2)

                except Exception as e:
                    print(f"ERROR in {exp_name}: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    from collections import defaultdict
    summary = defaultdict(list)
    for r in all_results:
        summary[(r["model"], r["n_samples"], r.get("augment"))].append(r["accuracy"])

    print(f"\n{'Model':<20} {'Aug':<10} {'K':<8} {'Mean Acc':<10} {'Std':<10}")
    print("-" * 60)
    for (model, k, aug), accs in sorted(summary.items()):
        aug_str = aug if aug else "none"
        print(f"{model:<20} {aug_str:<10} {str(k):<8} {np.mean(accs):.4f}     ±{np.std(accs):.4f}")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
