"""
Baseline Evaluation Script

Runs zero-shot inference on a stratified sample of ISIC images using the base MedGemma model.
Evaluates accuracy, macro F1, and per-class F1, outputting results and a confusion matrix.
"""
import os
import json
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from app.inference import MedGemmaInference, CONDITIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot baseline evaluation for DermScreen AI.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ISIC dataset directory containing metadata.csv.")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of images to sample.")
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory for predictions and metrics.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(args.data_dir) / "metadata.csv"
    if not csv_path.exists():
        logger.error(f"Metadata file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter to known conditions if necessary
    df = df[df['label'].isin(CONDITIONS)].dropna(subset=['image_path', 'label'])
    
    n_samples = min(args.n_samples, len(df))
    logger.info(f"Taking a stratified sample of {n_samples} images from {len(df)} total.")
    
    if len(df) > n_samples:
        _, sample_df = train_test_split(df, test_size=n_samples, stratify=df['label'], random_state=42)
    else:
        sample_df = df.copy()
        
    model = MedGemmaInference()
    
    predictions = []
    ground_truth = []
    
    prompt = f"Classify this skin lesion strictly into one of the following {len(CONDITIONS)} categories: {', '.join(CONDITIONS)}. Respond with the category name only."
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Running Inference"):
        img_path = Path(args.data_dir) / row['image_path']
        if not img_path.exists():
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            # We bypass the JSON wrapper to force category generation only
            raw_output = model._run_inference(prompt, image=image, max_new_tokens=10).strip().lower()
            
            # Extract category name assuming model might add formatting
            pred = "unknown"
            for c in CONDITIONS:
                if c.replace("_", " ") in raw_output or c in raw_output:
                    pred = c
                    break
                    
            predictions.append(pred)
            ground_truth.append(row['label'])
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            predictions.append("error")
            ground_truth.append(row['label'])
            
    # Calculate metrics
    valid_idx = [i for i, p in enumerate(predictions) if p != "error"]
    y_true = [ground_truth[i] for i in valid_idx]
    y_pred = [predictions[i] for i in valid_idx]
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    
    report = classification_report(y_true, y_pred, target_names=CONDITIONS, output_dict=True, zero_division=0)
    
    # Save predictions
    results_df = sample_df.iloc[valid_idx].copy()
    results_df['prediction'] = y_pred
    results_df.to_csv(output_dir / "zero_shot_predictions.csv", index=False)
    
    # Save metrics
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report
    }
    with open(output_dir / "zero_shot_summary.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CONDITIONS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CONDITIONS, yticklabels=CONDITIONS, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Zero-Shot MedGemma Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "zero_shot_confusion_matrix.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
