"""
DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
Task 2: Dual CNN Architecture (EfficientNet-B0 Training & Checkpoint)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Ensure local ml module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.dataset import PADUFESDataset, build_image_index, compute_class_weights, DIAGNOSTIC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from ml.models import build_efficientnet_b0, FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train DermAI EfficientNet-B0 Classifier")
    parser.add_argument("--metadata", type=str, default="metadata_with_folds.csv", help="Path to metadata with folds/splits")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--loss_type", type=str, default="weighted_ce", choices=["weighted_ce", "focal"], help="Loss function type")
    parser.add_argument("--output_dir", type=str, default="saved_models", help="Output directory for model checkpoints")
    parser.add_argument("--subset", type=int, default=None, help="Optional subset of training images for fast CPU fine-tuning")
    return parser.parse_args()


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (images, targets, _) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if (step + 1) % 10 == 0 or (step + 1) == len(loader):
            print(f"    Batch [{step+1}/{len(loader)}] Loss: {loss.item():.4f} Running Acc: {correct/total*100:.2f}%")

    return total_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    return val_loss, val_acc, np.array(all_targets), np.array(all_preds), np.array(all_probs)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Load metadata
    if not os.path.exists(args.metadata):
        if os.path.exists("data/" + args.metadata):
            args.metadata = "data/" + args.metadata
        else:
            raise FileNotFoundError(f"Metadata file {args.metadata} not found.")

    df = pd.read_csv(args.metadata)
    img_index = build_image_index()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if args.subset and args.subset < len(train_df):
        print(f"[*] Subsetting training set to {args.subset} samples for fast fine-tuning.")
        train_df = train_df.sample(n=args.subset, random_state=42).reset_index(drop=True)

    print(f"[*] Train set: {len(train_df)} | Val set: {len(val_df)} | Test set: {len(test_df)}")

    train_dataset = PADUFESDataset(train_df, img_index, is_training=True)
    val_dataset = PADUFESDataset(val_df, img_index, is_training=False)
    test_dataset = PADUFESDataset(test_df, img_index, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_weights = compute_class_weights(train_df, device)
    if args.loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build EfficientNet-B0
    model = build_efficientnet_b0(pretrained=True, num_classes=len(DIAGNOSTIC_CLASSES), dropout_rate=0.4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_weights_path = os.path.join(args.output_dir, "efficientnet_b0.pth")
    training_history = []

    print(f"\n[*] Starting EfficientNet-B0 training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch} Results | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | Time: {epoch_duration:.1f}s")

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_architecture": "EfficientNet-B0",
                "state_dict": model.state_dict(),
                "classes": DIAGNOSTIC_CLASSES,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, best_weights_path)
            print(f"  [+] Checkpoint saved to {best_weights_path} (Val Loss: {val_loss:.4f})")

    if not os.path.exists(best_weights_path):
        torch.save({
            "model_architecture": "EfficientNet-B0",
            "state_dict": model.state_dict(),
            "classes": DIAGNOSTIC_CLASSES,
            "epoch": args.epochs,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, best_weights_path)

    total_time = time.time() - start_time
    print(f"\n[*] Training completed in {total_time/60:.2f} minutes.")

    # Evaluate best model
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("\n" + "="*70)
    print("TASK 2 DELIVERABLE: EFFICIENTNET-B0 TEST SET CLASSIFICATION EVALUATION")
    print("="*70)
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(classification_report(y_true, y_pred, target_names=DIAGNOSTIC_CLASSES, digits=4, zero_division=0))
    print(f"Test Accuracy:    {test_acc*100:.2f}% | Macro F1-Score: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
