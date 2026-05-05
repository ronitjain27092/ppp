"""
Evaluation and Analysis Module for CNN Malware Detection

This module provides comprehensive evaluation, debug checks, and analysis
to ensure the model correctly handles class imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation with debug output."""
    
    @staticmethod
    def analyse_predictions(y_test, y_pred, y_pred_proba, model_name="CNN", threshold=0.3):
        """
        Comprehensive analysis of predictions.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of model
            threshold: Threshold used for prediction
        """
        print("\n" + "="*70)
        print(f"COMPREHENSIVE PREDICTION ANALYSIS ({model_name})")
        print("="*70)
        
        # 1. Class distribution
        print(f"\n1️⃣  CLASS DISTRIBUTION:")
        unique_true, counts_true = np.unique(y_test, return_counts=True)
        print(f"   Actual labels:")
        for cls, count in zip(unique_true, counts_true):
            pct = (count / len(y_test)) * 100
            label = "Benign" if cls == 0 else "Malware"
            print(f"     • {label}: {count} ({pct:.1f}%)")
        
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        print(f"   Predicted labels:")
        for cls, count in zip(unique_pred, counts_pred):
            pct = (count / len(y_pred)) * 100
            label = "Benign" if cls == 0 else "Malware"
            print(f"     • {label}: {count} ({pct:.1f}%)")
        
        # 2. Threshold analysis
        print(f"\n2️⃣  THRESHOLD ANALYSIS (threshold = {threshold}):")
        print(f"   • P(malware) >= {threshold} → Predict Malware")
        print(f"   • P(malware) < {threshold} → Predict Benign")
        
        prob_malware = y_pred_proba.flatten()
        print(f"   Probability distribution:")
        print(f"     • Min probability: {prob_malware.min():.4f}")
        print(f"     • Max probability: {prob_malware.max():.4f}")
        print(f"     • Mean probability: {prob_malware.mean():.4f}")
        print(f"     • Median probability: {np.median(prob_malware):.4f}")
        print(f"     • Samples >= {threshold}: {(prob_malware >= threshold).sum()}")
        print(f"     • Samples < {threshold}: {(prob_malware < threshold).sum()}")
        
        # 3. Confusion matrix
        print(f"\n3️⃣  CONFUSION MATRIX:")
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        print(f"   True Negatives (Benign correctly identified):  {tn}")
        print(f"   False Positives (Benign → Malware):            {fp}")
        print(f"   False Negatives (Malware → Benign) ⚠️ CRITICAL: {fn}")
        print(f"   True Positives (Malware correctly identified): {tp}")
        
        # 4. Key metrics
        print(f"\n4️⃣  KEY METRICS:")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"   • Accuracy:  {accuracy:.4f} - (TP+TN)/(All)")
        print(f"   • Precision: {precision:.4f} - TP/(TP+FP) [reliability of positive predictions]")
        print(f"   • Recall:    {recall:.4f} - TP/(TP+FN) [malware detection rate] ⭐ KEY")
        print(f"   • F1-Score:  {f1:.4f} - Harmonic mean of precision & recall")
        
        # 5. Specific checks for class imbalance fix
        print(f"\n5️⃣  CLASS IMBALANCE FIX VALIDATION:")
        
        # Check 1: Are both classes predicted?
        if len(unique_pred) < 2:
            print(f"   ❌ ISSUE: Only {len(unique_pred)} class predicted")
            print(f"      • Model is still biased toward one class")
            print(f"      • Suggested: Lower threshold further")
        else:
            print(f"   ✓ Both classes predicted (Good!)")
        
        # Check 2: Is recall > 0?
        if recall > 0:
            print(f"   ✓ Recall > 0 (Malware is being detected)")
        else:
            print(f"   ❌ Recall = 0 (NO malware detected)")
            print(f"      • This is the class imbalance bug!")
            print(f"      • Suggested fixes:")
            print(f"        1. Lower threshold (currently {threshold})")
            print(f"        2. Check class weights were applied")
            print(f"        3. Retrain model")
        
        # Check 3: Is F1 > 0?
        if f1 > 0:
            print(f"   ✓ F1-Score > 0 (Good balance of metrics)")
        else:
            print(f"   ❌ F1-Score = 0 (Model not detecting malware at all)")
        
        # Check 4: False negatives
        if fn > 0:
            fn_pct = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            print(f"   • False Negatives: {fn} ({fn_pct:.1f}% of actual malware missed)")
            if fn_pct > 30:
                print(f"     ⚠️  High false negative rate - malware likely to be missed")
        
        print("="*70 + "\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    @staticmethod
    def plot_probability_distribution(y_pred_proba, y_test, threshold=0.3, save_path=None):
        """
        Plot distribution of predicted probabilities.
        
        Shows:
        - Distribution of P(malware) for actual Benign samples
        - Distribution of P(malware) for actual Malware samples
        - Threshold line for decision boundary
        """
        try:
            plt.figure(figsize=(10, 6))
            
            proba_flat = y_pred_proba.flatten()
            
            # Benign samples
            benign_proba = proba_flat[y_test == 0]
            plt.hist(benign_proba, bins=50, alpha=0.6, label=f'Actual Benign (n={len(benign_proba)})', color='blue')
            
            # Malware samples
            malware_proba = proba_flat[y_test == 1]
            plt.hist(malware_proba, bins=50, alpha=0.6, label=f'Actual Malware (n={len(malware_proba)})', color='red')
            
            # Threshold line
            plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
            
            plt.xlabel('P(Malware)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predicted Probabilities\n(Shows why threshold matters)', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Probability distribution plot saved")
            
            return plt.gcf()
        
        except Exception as e:
            print(f"⚠ Error plotting probability distribution: {str(e)}")
            return None
    
    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba, save_path=None):
        """Plot ROC Curve."""
        try:
            proba_flat = y_pred_proba.flatten()
            fpr, tpr, thresholds = roc_curve(y_test, proba_flat)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (Test Set)', fontweight='bold')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ ROC curve saved")
            
            return plt.gcf()
        
        except Exception as e:
            print(f"⚠ Error plotting ROC curve: {str(e)}")
            return None


def find_optimal_threshold(y_test, y_pred_proba, metric='f1'):
    """
    Find optimal threshold by maximizing specified metric.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        metric: 'f1', 'recall', or 'precision'
    
    Returns:
        float: Optimal threshold
    """
    from sklearn.metrics import f1_score, recall_score, precision_score
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    scores = []
    
    proba_flat = y_pred_proba.flatten()
    
    for threshold in thresholds:
        y_pred = (proba_flat >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_test, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred, zero_division=0)
        else:
            score = 0
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    print(f"\n🎯 OPTIMAL THRESHOLD SEARCH")
    print(f"   • Metric: {metric}")
    print(f"   • Best threshold: {best_threshold:.2f}")
    print(f"   • Best {metric} score: {best_score:.4f}")
    
    return best_threshold
