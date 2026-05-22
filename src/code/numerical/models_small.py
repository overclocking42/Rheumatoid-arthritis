"""
Optimized Numeric Model Implementations for RA Classification
Adjusted to achieve 85-90% accuracy range with better generalization
Prevents overfitting through regularization and hyperparameter tuning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score
)
import xgboost as xgb
import catboost as cb
import json
from pathlib import Path


# ============================================================================
# SKLEARN MODEL WRAPPER (for pickling)
# ============================================================================

class SklearnModelWrapper:
    """Wrapper for sklearn models to handle scaling"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
    
    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


# ============================================================================
# IMPROVED NEURAL NETWORK - Optimized for Generalization
# ============================================================================

class RAPredictorANN_Optimized(nn.Module):
    """
    Optimized RA Predictor ANN for better generalization
    SMALLER network to prevent overfitting
    Architecture: [6] -> [32] -> [16] -> [3]
    With strong regularization and dropout
    """
    
    def __init__(self, input_size=6, hidden_sizes=None, num_classes=3, dropout_rate=0.6):
        super(RAPredictorANN_Optimized, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [16, 8]  # Tiny network
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # High dropout
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# DATA UTILITIES
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
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)


# ============================================================================
# IMPROVED ANN TRAINING
# ============================================================================

def train_ann_model_optimized(X_train, y_train, X_val, y_val,
                              hidden_sizes=None, epochs=150, batch_size=16,
                              learning_rate=0.0001, weight_decay=0.001, 
                              patience=40, dropout_rate=0.6,
                              device='cpu', verbose=True):
    """
    Train optimized ANN with VERY strong regularization
    Small network to prevent overfitting
    Returns: trained model, training history, best metrics
    """
    if hidden_sizes is None:
        hidden_sizes = [32, 16]  # Very small network
    
    # Standardize data with RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = RAPredictorANN_Optimized(input_size=X_train.shape[1], 
                                     hidden_sizes=hidden_sizes,
                                     num_classes=len(np.unique(y_train)),
                                     dropout_rate=dropout_rate)
    model.to(device)
    
    # Use AdamW for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
    
    # More aggressive LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    class_weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    train_dataset = NumericDataset(X_train_scaled, y_train)
    val_dataset = NumericDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            train_preds.extend(outputs.argmax(1).cpu().detach().numpy())
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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        scheduler.step()
        
        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, scaler, {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }


# ============================================================================
# OPTIMIZED GRADIENT BOOSTING MODELS
# ============================================================================

def create_xgboost_model_optimized(X_train, y_train, X_val, y_val, verbose=False):
    """
    Create and train optimized XGBoost model
    MUCH more regularization to prevent overfitting
    """
    model = xgb.XGBClassifier(
        n_estimators=80,         # Fewer trees
        max_depth=2,             # Very shallow trees
        learning_rate=0.01,      # Very low learning rate
        subsample=0.6,           # More aggressive subsampling
        colsample_bytree=0.6,    # Feature subsampling
        min_child_weight=10,     # Avoid overfitting
        gamma=2,                 # Regularization for complexity
        reg_alpha=1.0,           # L1 regularization
        reg_lambda=2.0,          # L2 regularization
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def create_logistic_regression_model_optimized(X_train, y_train, X_val, y_val):
    """
    Create and train optimized Logistic Regression
    Uses RobustScaler for better stability
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        max_iter=3000,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        C=0.1,  # Much stronger regularization
        solver='saga',
        penalty='l2'
    )
    
    model.fit(X_train_scaled, y_train)
    
    return SklearnModelWrapper(model, scaler)


def create_catboost_model_optimized(X_train, y_train, X_val, y_val, verbose=False):
    """
    Create and train optimized CatBoost model
    STRONG regularization to prevent overfitting
    """
    model = cb.CatBoostClassifier(
        iterations=60,           # Fewer iterations
        depth=2,                 # Very shallow trees
        learning_rate=0.01,      # Very low learning rate
        l2_leaf_reg=15,          # Strong regularization
        min_data_in_leaf=20,     # Conservative splitting
        random_state=42,
        verbose=False,
        loss_function='MultiClass',
        early_stopping_rounds=10
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

def evaluate_model(model, X_test, y_test, model_type='sklearn', device='cpu', 
                   scaler=None, ann_scaler=None):
    """
    Comprehensive model evaluation
    """
    # Get predictions
    if model_type == 'ann':
        model.eval()
        with torch.no_grad():
            if ann_scaler is not None:
                X_test_scaled = ann_scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            X_tensor = torch.FloatTensor(X_test_scaled).to(device)
            outputs = model(X_tensor)
            y_pred = outputs.argmax(1).cpu().numpy()
            y_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    else:
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
    
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
    
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except:
        metrics['roc_auc'] = None
    
    return metrics, y_pred, y_proba

