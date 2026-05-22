#!/usr/bin/env python3
"""
Professional Numeric Model Training with 5-Fold Cross-Validation
Trains 4 candidates: XGBoost, ANN, Logistic Regression, CatBoost
Evaluates on held-out test set with comprehensive metrics
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.models_small import (
    train_ann_model_optimized, create_xgboost_model_optimized,
    create_catboost_model_optimized, evaluate_model
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': Path('data/processed/splits'),
    'models_dir': Path('models/numeric'),
    'results_dir': Path('reports/numeric'),
    'seed': 42,
    'device': 'cpu',
    'verbose': True,
    'ann_config': {
        'hidden_sizes': [256, 128, 64],
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 30
    }
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Create necessary directories"""
    CONFIG['models_dir'].mkdir(parents=True, exist_ok=True)
    CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ready: {CONFIG['models_dir']}, {CONFIG['results_dir']}")


def load_data(split_name):
    """Load train/val/test data"""
    prefix = CONFIG['data_dir'] / f'numeric_{split_name}'
    df = pd.read_csv(f'{prefix}.csv')
    
    # Feature columns (all except target)
    feature_cols = [col for col in df.columns if col != 'label' and col != 'SampleID']
    X = df[feature_cols].values
    y = df['label'].values
    
    return X, y, feature_cols


# ============================================================================
# CROSS-VALIDATION TRAINING
# ============================================================================

def train_single_candidate(candidate_name, fold_num, X_fold_train, y_fold_train,
                          X_fold_val, y_fold_val, X_test, y_test):
    """
    Train a single candidate on a fold
    Returns: trained model, train metrics, val metrics
    """
    
    if candidate_name == 'ANN':
        print(f"  [{candidate_name}] Fold {fold_num}: Training ANN [256,128,64]...", end=' ', flush=True)
        model, history, meta = train_ann_model(
            X_fold_train, y_fold_train,
            X_fold_val, y_fold_val,
            **CONFIG['ann_config'],
            device=CONFIG['device'],
            verbose=False
        )
        print(f"✓ (best epoch: {meta['best_epoch']+1})")
        
    elif candidate_name == 'XGBoost':
        print(f"  [{candidate_name}] Fold {fold_num}: Training...", end=' ', flush=True)
        model = create_xgboost_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        print("✓")
        
    elif candidate_name == 'LogReg':
        print(f"  [{candidate_name}] Fold {fold_num}: Training...", end=' ', flush=True)
        model = create_logistic_regression_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        print("✓")
        
    elif candidate_name == 'CatBoost':
        print(f"  [{candidate_name}] Fold {fold_num}: Training...", end=' ', flush=True)
        model = create_catboost_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
        print("✓")
    
    # Evaluate on validation fold
    model_type = 'ann' if candidate_name == 'ANN' else 'sklearn'
    val_metrics, _, _ = evaluate_model(
        model, X_fold_val, y_fold_val, model_type=model_type, device=CONFIG['device']
    )
    
    # Evaluate on test set
    test_metrics, test_pred, test_proba = evaluate_model(
        model, X_test, y_test, model_type=model_type, device=CONFIG['device']
    )
    
    return model, val_metrics, test_metrics, test_pred, test_proba


def train_cv_fold(fold_num, feature_cols):
    """
    Train all candidates on a single fold
    Returns: dict with results for all candidates
    """
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION FOLD {fold_num}")
    print(f"{'='*70}")
    
    # Load fold data
    fold_prefix = CONFIG['data_dir'] / f'numeric_cv_fold{fold_num}'
    
    X_train = pd.read_csv(f'{fold_prefix}_train.csv')[feature_cols].values
    y_train = pd.read_csv(f'{fold_prefix}_train.csv')['label'].values
    
    X_val = pd.read_csv(f'{fold_prefix}_val.csv')[feature_cols].values
    y_val = pd.read_csv(f'{fold_prefix}_val.csv')['label'].values
    
    # Load test set (same for all folds)
    X_test, y_test, _ = load_data('test')
    
    print(f"Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples | Test: {X_test.shape[0]} samples")
    
    # Train all candidates
    fold_results = {}
    candidates = ['ANN', 'XGBoost', 'LogReg', 'CatBoost']
    
    for candidate in candidates:
        try:
            model, val_met, test_met, test_pred, test_proba = train_single_candidate(
                candidate, fold_num, X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            fold_results[candidate] = {
                'model': model,
                'val_metrics': val_met,
                'test_metrics': test_met,
                'test_pred': test_pred,
                'test_proba': test_proba
            }
            
            # Print test metrics
            acc = test_met['accuracy']
            f1 = test_met['macro_f1']
            print(f"    → Test Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
            
        except Exception as e:
            print(f"\n    ✗ FAILED: {str(e)}")
            fold_results[candidate] = None
    
    return fold_results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PROFESSIONAL NUMERIC MODEL TRAINING")
    print("Cross-Validation with 5 Folds")
    print("="*70)
    
    # Setup
    setup_directories()
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    # Load feature columns once
    _, _, feature_cols = load_data('train')
    print(f"\nFeatures ({len(feature_cols)}): {', '.join(feature_cols)}")
    
    # Train on all folds
    all_fold_results = {}
    
    for fold_num in range(1, 6):
        fold_results = train_cv_fold(fold_num, feature_cols)
        all_fold_results[fold_num] = fold_results
    
    # Aggregate results across folds
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    summary_stats = {}
    
    for candidate in ['ANN', 'XGBoost', 'LogReg', 'CatBoost']:
        print(f"\n{candidate}:")
        
        test_accuracies = []
        test_f1s = []
        
        for fold_num in range(1, 6):
            if all_fold_results[fold_num][candidate] is not None:
                test_met = all_fold_results[fold_num][candidate]['test_metrics']
                test_accuracies.append(test_met['accuracy'])
                test_f1s.append(test_met['macro_f1'])
        
        if test_accuracies:
            mean_acc = np.mean(test_accuracies)
            std_acc = np.std(test_accuracies)
            mean_f1 = np.mean(test_f1s)
            std_f1 = np.std(test_f1s)
            
            print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"  Folds: {test_accuracies}")
            
            summary_stats[candidate] = {
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'f1_mean': mean_f1,
                'f1_std': std_f1,
                'accuracy_per_fold': test_accuracies,
                'f1_per_fold': test_f1s
            }
    
    # Save summary
    summary_path = CONFIG['results_dir'] / 'cv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\n✓ Summary saved to {summary_path}")
    
    # Save models and results
    print(f"\nSaving models and results...")
    for fold_num in range(1, 6):
        fold_dir = CONFIG['models_dir'] / f'fold{fold_num}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        for candidate in ['ANN', 'XGBoost', 'LogReg', 'CatBoost']:
            if all_fold_results[fold_num][candidate] is not None:
                result = all_fold_results[fold_num][candidate]
                
                if candidate == 'ANN':
                    # Save PyTorch model
                    model_path = fold_dir / f'{candidate.lower()}_model.pth'
                    torch.save(result['model'].state_dict(), model_path)
                else:
                    # Save sklearn models
                    model_path = fold_dir / f'{candidate.lower()}_model.pkl'
                    with open(model_path, 'wb') as f:
                        pickle.dump(result['model'], f)
    
    print(f"✓ Models saved to {CONFIG['models_dir']}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nResults summary available at: {summary_path}")
    print(f"Trained models available at: {CONFIG['models_dir']}")


if __name__ == '__main__':
    main()
