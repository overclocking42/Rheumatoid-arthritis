#!/usr/bin/env python3
"""
ResNet50 Training with Augmentation - M4 GPU Optimized
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Setup
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")


class XRayDataset(Dataset):
    """Hand X-ray dataset with augmentation"""
    def __init__(self, csv_path: str, size: int = 224, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.size = size
        self.train = train
        self.label_map = {"non_erosive": 0, "erosive": 1}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.float32)
        
        # Load image
        img = self._load_img(path)
        x = self._preprocess(img).to(torch.float32)
        
        return x.to(torch.float32), label, path
    
    def _load_img(self, path):
        try:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                return Image.fromarray(arr)
        except:
            pass
        try:
            return Image.open(path)
        except:
            return Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
    
    def _preprocess(self, img):
        img = img.convert("L")
        arr = np.array(img).astype(np.float32)
        
        # Percentile clipping
        lo, hi = np.percentile(arr, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        
        # To tensor and resize
        t = torch.from_numpy(arr).unsqueeze(0)
        t = F.interpolate(t.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)
        
        # Normalize
        t = (t - 0.5) / 0.25
        
        # Apply augmentation if training
        if self.train:
            # Random flip
            if np.random.rand() < 0.5:
                t = torch.flip(t, dims=[-1])
            # Random rotation (approximate)
            if np.random.rand() < 0.5:
                angle = np.random.randint(-15, 15)
                t = transforms.functional.rotate(t, angle)
            # Random brightness
            if np.random.rand() < 0.5:
                t = t * (0.8 + 0.4 * np.random.rand())
        
        # Repeat to 3 channels
        t = t.repeat(3, 1, 1)
        return t


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


def get_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts().to_dict()
    n_ne = counts.get('non_erosive', 1)
    n_e = counts.get('erosive', 1)
    total = n_ne + n_e
    
    weights = []
    for label in df['label']:
        if label == 'erosive':
            weights.append(total / (2 * n_e))
        else:
            weights.append(total / (2 * n_ne))
    return torch.tensor(weights, dtype=torch.float32)


def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 1))
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        x = x.to(device, dtype=torch.float32)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        ps.extend(p)
        ys.extend(y.numpy())
    
    ys, ps = np.array(ys), np.array(ps)
    pred = (ps >= 0.5).astype(int)
    
    acc = accuracy_score(ys, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, pred, average=None, zero_division=0)
    mf1 = (f1[0] + f1[1]) / 2
    
    try:
        roc = roc_auc_score(ys, ps)
    except:
        roc = np.nan
    
    cm = confusion_matrix(ys, pred, labels=[0, 1]).tolist()
    
    return {
        "accuracy": float(acc),
        "precision": {"non_erosive": float(prec[0]), "erosive": float(prec[1])},
        "recall": {"non_erosive": float(rec[0]), "erosive": float(rec[1])},
        "f1": {"non_erosive": float(f1[0]), "erosive": float(f1[1])},
        "macro_f1": float(mf1),
        "roc_auc": float(roc),
        "confusion_matrix": cm,
        "y_true": ys.tolist(),
        "y_prob": ps.tolist(),
    }


def train_epoch(model, loader, opt, device, loss_fn):
    model.train()
    total_loss = 0
    n = 0
    for x, y, _ in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def plot_results(metrics, reports_dir, model_name):
    ys = np.array(metrics['y_true'])
    ps = np.array(metrics['y_prob'])
    pred = (ps >= 0.5).astype(int)
    
    # ROC
    try:
        fpr, tpr, _ = roc_curve(ys, ps)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={metrics.get('roc_auc', 0):.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("FPR"), plt.ylabel("TPR")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()
        plt.savefig(f"{reports_dir}/roc_{model_name}.png", dpi=100, bbox_inches="tight")
        plt.close()
    except:
        pass
    
    # Confusion Matrix
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Greens")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xticks([0, 1], ["Non-Erosive", "Erosive"])
    plt.yticks([0, 1], ["Non-Erosive", "Erosive"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.ylabel("True"), plt.xlabel("Predicted")
    plt.savefig(f"{reports_dir}/confusion_matrix_{model_name}.png", dpi=100, bbox_inches="tight")
    plt.close()


def train_model(model_name, max_epochs=30):
    print(f"\n{'='*70}\nTraining {model_name.upper()}\n{'='*70}")
    
    base = "/Users/joyboy/Documents/cursor/project-root"
    splits_dir = f"{base}/data/raw_data/imaging/RAM-W600/splits"
    models_dir = f"{base}/models"
    reports_dir = f"{base}/reports/image"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Datasets
    train_ds = XRayDataset(f"{splits_dir}/train.csv", train=True)
    val_ds = XRayDataset(f"{splits_dir}/val.csv", train=False)
    test_ds = XRayDataset(f"{splits_dir}/test.csv", train=False)
    
    # DataLoaders with weighted sampler
    weights = get_class_weights(f"{splits_dir}/train.csv")
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Model, loss, optimizer
    model = build_resnet50().to(DEVICE)
    criterion = FocalLoss(0.25, 2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    best_f1 = 0
    patience = 0
    max_patience = 10
    
    print(f"Training with WeightedRandomSampler + Focal Loss")
    print(f"Batch size: 16, Learning rate: 1e-4, Max epochs: {max_epochs}\n")
    
    for epoch in range(max_epochs):
        loss = train_epoch(model, train_loader, optimizer, DEVICE, criterion)
        val_metrics = evaluate(model, val_loader, DEVICE)
        f1_erosive = val_metrics['f1']['erosive']
        
        scheduler.step()
        
        if f1_erosive > best_f1:
            best_f1 = f1_erosive
            patience = 0
            torch.save(model.state_dict(), f"{models_dir}/{model_name}.pth")
            print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | Val F1 (erosive): {f1_erosive:.4f} ✓")
        else:
            patience += 1
            print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | Val F1 (erosive): {f1_erosive:.4f}")
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate on test
    print(f"\nEvaluating on test set...")
    model.load_state_dict(torch.load(f"{models_dir}/{model_name}.pth"))
    test_metrics = evaluate(model, test_loader, DEVICE)
    
    # Save metrics
    with open(f"{reports_dir}/metrics_{model_name}.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Plot
    plot_results(test_metrics, reports_dir, model_name)
    
    print(f"\n{model_name.upper()} Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (erosive): {test_metrics['f1']['erosive']:.4f}")
    print(f"  F1 (non-erosive): {test_metrics['f1']['non_erosive']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    
    return test_metrics


if __name__ == "__main__":
    print("="*70)
    print("M4 GPU-OPTIMIZED RESNET50 TRAINING WITH AUGMENTATION")
    print("="*70)
    
    resnet_metrics = train_model("resnet50", max_epochs=30)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nResNet50:")
    print(f"  Test Accuracy: {resnet_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1: {resnet_metrics['macro_f1']:.4f}")
