"""
Professional Numeric Model Implementations for RA Classification
Includes: XGBoost, ANN, Logistic Regression, CatBoost
All with proper preprocessing, validation, and evaluation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score
)
import xgboost as xgb
import catboost as cb
import json
from pathlib import Path


# ============================================================================
# NEURAL NETWORK MODEL - Updated Architecture [256, 128, 64]
# ============================================================================

class RAPredictorANN(nn.Module):
    """
    Professional Rheumatoid Arthritis Predictor ANN
    Architecture: [7] -> [256] -> [128] -> [64] -> [3]
    With batch normalization, dropout, and L2 regularization
    """
    
    def __init__(self, input_size=7, hidden_sizes=None, num_classes=3, dropout_rate=0.3):
        super(RAPredictorANN, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]  # Updated architecture
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class NumericDataset(TensorDataset):
    """PyTorch Dataset for numeric RA data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.LongTensor(y) if not isinstance(y, torch.Tensor) else y
        assert len(self.X) == len(self.y), "X and y must have same length"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_class_weights(y):
    """Compute balanced class weights for imbalanced data"""
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    # Normalize to sum to 1
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)


# ============================================================================
# FOCAL LOSS (for imaging, but included here for reference)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# TRAINING FUNCTION FOR ANN
# ============================================================================

def train_ann_model(X_train, y_train, X_val, y_val,
                   hidden_sizes=None, epochs=100, batch_size=32,
                   learning_rate=0.001, weight_decay=1e-4, patience=30,
                   device='cpu', verbose=True):
    """
    Train ANN model with professional validation and early stopping
    
    Returns: trained model, training history, best metrics
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]
    
    # Create model
    model = RAPredictorANN(input_size=X_train.shape[1], 
                           hidden_sizes=hidden_sizes,
                           num_classes=len(np.unique(y_train)))
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Weighted loss for class imbalance
    class_weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Create data loaders
    train_dataset = NumericDataset(X_train, y_train)
    val_dataset = NumericDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_f1': history['train_f1'][-1],
        'final_val_f1': history['val_f1'][-1]
    }


# ============================================================================
# MODEL FACTORY & TRAINING WRAPPERS
# ============================================================================

def create_xgboost_model(X_train, y_train, X_val, y_val, verbose=False):
    """Create and train XGBoost model"""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=0
    )
    
    return model


def create_logistic_regression_model(X_train, y_train, X_val, y_val):
    """Create and train Logistic Regression model"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Create wrapper that handles scaling
    class LogRegWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        
        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))
        
        def predict_proba(self, X):
            return self.model.predict_proba(self.scaler.transform(X))
    
    return LogRegWrapper(model, scaler)


def create_catboost_model(X_train, y_train, X_val, y_val, verbose=False):
    """Create and train CatBoost model"""
    model = cb.CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False,
        loss_function='MultiClass'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_model(model, X_test, y_test, model_type='sklearn', device='cpu'):
    """
    Comprehensive model evaluation
    Returns: metrics dict with per-class breakdown
    """
    # Get predictions
    if model_type == 'ann':
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_tensor)
            y_pred = outputs.argmax(1).cpu().numpy()
            y_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'per_class': {}
    }
    
    # Per-class metrics
    classes = np.unique(y_test)
    class_names = {0: 'Healthy', 1: 'Seropositive', 2: 'Seronegative'}
    
    for cls in classes:
        mask = y_test == cls
        if mask.sum() > 0:
            y_pred_binary = (y_pred == cls).astype(int)
            y_test_binary = (y_test == cls).astype(int)
            
            metrics['per_class'][class_names.get(cls, f'Class_{cls}')] = {
                'precision': precision_score(y_test_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_test_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_test_binary, y_pred_binary, zero_division=0),
                'support': mask.sum()
            }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    # ROC-AUC (one-vs-rest for multiclass)
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except:
        metrics['roc_auc'] = None
    
    return metrics, y_pred, y_proba
