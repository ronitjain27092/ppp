"""
ENHANCED EVALUATION & VISUALIZATION MODULE
===========================================

Provides comprehensive, publication-ready visualizations to detect data leakage.

Corrected ROC Curves - They should show the realistic diagonal or curve,
NOT a perfect square (which would indicate overfitting/leakage).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ComprehensiveEvaluator:
    """Enhanced evaluation with realistic visualizations."""
    
    def __init__(self):
        self.results = {}
    
    # =========================================================================
    # CORRECTED ROC CURVES (Should NOT be perfect square)
    # =========================================================================
    def plot_roc_curves_correct(self, y_test, y_pred_proba_dict, model_names=None):
        """
        Plot ROC curves correctly using predict_proba.
        
        CRITICAL FIXES:
        - Use y_pred_proba (probabilities), NOT y_pred (hard predictions)
        - Should show realistic curve, NOT perfect square
        - Plot on single figure for comparison
        
        Args:
            y_test: True labels
            y_pred_proba_dict: Dict of {model_name: probabilities}
            model_names: Model names for legend
        
        Returns:
            fig: Matplotlib figure
        """
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_pred_proba_dict)))
        
        for (model_name, y_pred_proba), color in zip(y_pred_proba_dict.items(), colors):
            # CRITICAL: Use probabilities for ROC
            y_pred_proba = y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba
            y_pred_proba = y_pred_proba.flatten()
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=color, lw=2.5,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)',
               alpha=0.7)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves Comparison\n(Realistic curves should NOT be a perfect square)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # PRECISION-RECALL CURVES (Better for Imbalanced Data)
    # =========================================================================
    def plot_precision_recall_curves(self, y_test, y_pred_proba_dict):
        """
        Precision-Recall curves are better for imbalanced datasets.
        
        For perfect leakage:
        - Curve would be near the top-right corner
        - F1 score (harmonic mean) would be near 1.0
        
        For realistic models:
        - Curve shows tradeoff between precision and recall
        - Baseline is at the proportion of positive class
        """
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_pred_proba_dict)))
        baseline = (y_test == 1).sum() / len(y_test)
        
        for (model_name, y_pred_proba), color in zip(y_pred_proba_dict.items(), colors):
            y_pred_proba = y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba
            y_pred_proba = y_pred_proba.flatten()
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = np.mean(precision) if len(precision) > 0 else 0
            
            ax.plot(recall, precision, color=color, lw=2.5,
                   label=f'{model_name} (AP = {ap:.3f})')
        
        # Baseline (random classifier = proportion of positives)
        ax.axhline(y=baseline, color='k', linestyle='--', lw=2,
                  label=f'Baseline (Positive Ratio = {baseline:.3f})', alpha=0.7)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves\n(Better for imbalanced datasets)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # CALIBRATION CURVE (Shows if probabilities are honest)
    # =========================================================================
    def plot_calibration_curve(self, y_test, y_pred_proba_dict, n_bins=10):
        """
        Calibration curve shows if model probabilities are reliable.
        
        For leakage: Curve would be near top-left (probabilities unrealistically confident)
        For honest model: Curve should follow diagonal
        """
        from sklearn.calibration import calibration_curve
        
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_pred_proba_dict)))
        
        for (model_name, y_pred_proba), color in zip(y_pred_proba_dict.items(), colors):
            y_pred_proba = y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba
            y_pred_proba = y_pred_proba.flatten()
            
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)
            
            ax.plot(prob_pred, prob_true, color=color, marker='o', linewidth=2.5,
                   markersize=8, label=model_name)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration', alpha=0.7)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax.set_title('Calibration Curves\n(Probabilities should be honest)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # THRESHOLD ANALYSIS
    # =========================================================================
    def plot_threshold_analysis(self, y_test, y_pred_proba, model_name="Model"):
        """
        Show how metrics change with prediction threshold.
        
        For leakage: Metrics stay near 1.0 across wide threshold range
        For honest model: Metrics show clear tradeoff curve
        """
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        y_pred_proba = y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba
        y_pred_proba = y_pred_proba.flatten()
        
        thresholds = np.linspace(0, 1, 101)
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            
            if len(np.unique(y_pred)) < 2:  # All same class
                metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['precision'].append(0 if thresh < 0.5 else 1)
                metrics['recall'].append(0 if thresh > 0.5 else 1)
                metrics['f1'].append(0)
            else:
                metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for metric, color in zip(['accuracy', 'precision', 'recall', 'f1'],
                                 ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']):
            ax.plot(thresholds, metrics[metric], label=metric.capitalize(),
                   linewidth=2.5, color=color, marker='o', markersize=4, alpha=0.8)
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Decision Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Threshold Analysis - {model_name}\n'
                    '(Smooth curves indicate honest model; flat high lines indicate leakage)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Default (0.5)')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # PROBABILITY DISTRIBUTION
    # =========================================================================
    def plot_probability_distribution(self, y_test, y_pred_proba, model_name="Model"):
        """
        Show distribution of predicted probabilities for each class.
        
        For leakage: Bimodal distribution at 0 and 1
        For honest model: Smooth bell-curves separated
        """
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        y_pred_proba = y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba
        y_pred_proba = y_pred_proba.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax = axes[0]
        benign_proba = y_pred_proba[y_test == 0]
        malware_proba = y_pred_proba[y_test == 1]
        
        ax.hist(benign_proba, bins=30, alpha=0.6, label='Benign (True)', color='green', edgecolor='black')
        ax.hist(malware_proba, bins=30, alpha=0.6, label='Malware (True)', color='red', edgecolor='black')
        ax.set_xlabel('Predicted Probability of Malware', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}: Probability Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # KDE
        ax = axes[1]
        try:
            from scipy import stats
            x_range = np.linspace(0, 1, 200)
            if len(benign_proba) > 1:
                kde_benign = stats.gaussian_kde(benign_proba)
                ax.plot(x_range, kde_benign(x_range), 'g-', linewidth=2.5, label='Benign')
            if len(malware_proba) > 1:
                kde_malware = stats.gaussian_kde(malware_proba)
                ax.plot(x_range, kde_malware(x_range), 'r-', linewidth=2.5, label='Malware')
        except:
            pass
        
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Default Threshold')
        ax.set_xlabel('Predicted Probability of Malware', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}: Probability Density', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # CONFUSION MATRICES
    # =========================================================================
    def plot_confusion_matrices(self, y_test, y_pred_dict):
        """
        Compare confusion matrices across models.
        
        For leakage: All matrices would be near-diagonal (few errors)
        For honest models: More spread in off-diagonal elements
        """
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        n_models = len(y_pred_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for (model_name, y_pred), ax in zip(y_pred_dict.items(), axes):
            y_pred = y_pred.values if hasattr(y_pred, 'values') else y_pred
            y_pred = y_pred.flatten()
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalize
            cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                       xticklabels=['Benign', 'Malware'],
                       yticklabels=['Benign', 'Malware'],
                       cbar_kws={'label': 'Percentage'})
            
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        fig.suptitle('Confusion Matrices (Normalized %)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # METRICS COMPARISON
    # =========================================================================
    def plot_metrics_comparison_table(self, results_dict):
        """
        Create detailed metrics comparison table as visual.
        
        Args:
            results_dict: {model_name: {metric_name: value, ...}, ...}
        """
        df = pd.DataFrame(results_dict).T.round(4)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Color cells based on values
        table = ax.table(cellText=df.values.astype(str),
                        colLabels=df.columns,
                        rowLabels=df.index,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15]*len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows with high metrics (potential leakage)
        for i, row in enumerate(df.values, 1):
            for j, val in enumerate(row):
                try:
                    val_float = float(val)
                    if val_float >= 0.99:
                        table[(i, j)].set_facecolor('#FFE699')  # Yellow (suspicious)
                    elif val_float >= 0.95:
                        table[(i, j)].set_facecolor('#FFC7CE')  # Light red
                except:
                    pass
            table[(i, 0)].set_facecolor('#E7E6E6')
        
        plt.title('Model Metrics Comparison\n(Yellow = Suspicious, Red = Potential Leakage)',
                 fontsize=13, fontweight='bold', pad=20)
        plt.subplots_adjust(left=0.25, top=0.85)
        
        return fig
    
    # =========================================================================
    # TRAIN VS TEST GAP
    # =========================================================================
    def plot_train_test_gap(self, train_results_dict, test_results_dict):
        """
        Visualize generalization gap (train - test performance).
        
        For leakage: Gap near zero
        For realistic: Gap 5-15% depending on dataset
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        models = list(train_results_dict.keys())
        
        gaps = {}
        for metric in metrics:
            gaps[metric] = []
            for model in models:
                train_val = train_results_dict.get(model, {}).get(metric, 0)
                test_val = test_results_dict.get(model, {}).get(metric, 0)
                gaps[metric].append(train_val - test_val)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, gaps[metric], width, label=metric.capitalize())
        
        ax.set_ylabel('Train - Test Value', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Generalization Gap (Train - Test)\n'
                    '(Near zero = potential leakage; 5-15% = realistic)',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='5% threshold')
        ax.axhline(y=0.15, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='15% threshold')
        
        plt.tight_layout()
        return fig


# =========================================================================
# HELPER FUNCTION
# =========================================================================
def create_all_evaluation_plots(y_test, y_pred_proba_dict, y_pred_dict,
                                train_results_dict=None, test_results_dict=None,
                                model_names=None, output_dir=None):
    """
    Create all evaluation plots at once.
    
    Args:
        y_test: True labels
        y_pred_proba_dict: {model_name: probabilities}
        y_pred_dict: {model_name: predictions}
        train_results_dict: {model_name: {metric: value}}
        test_results_dict: {model_name: {metric: value}}
        output_dir: Directory to save plots
    """
    evaluator = ComprehensiveEvaluator()
    figures = {}
    
    figures['roc_curves'] = evaluator.plot_roc_curves_correct(y_test, y_pred_proba_dict)
    figures['pr_curves'] = evaluator.plot_precision_recall_curves(y_test, y_pred_proba_dict)
    figures['calibration'] = evaluator.plot_calibration_curve(y_test, y_pred_proba_dict)
    figures['confusion_matrices'] = evaluator.plot_confusion_matrices(y_test, y_pred_dict)
    
    # Single model plots (use first model)
    if len(y_pred_proba_dict) > 0:
        first_model = list(y_pred_proba_dict.keys())[0]
        figures['probability_dist'] = evaluator.plot_probability_distribution(
            y_test, y_pred_proba_dict[first_model], first_model
        )
        figures['threshold_analysis'] = evaluator.plot_threshold_analysis(
            y_test, y_pred_proba_dict[first_model], first_model
        )
    
    if train_results_dict and test_results_dict:
        figures['train_test_gap'] = evaluator.plot_train_test_gap(
            train_results_dict, test_results_dict
        )
    
    # Save if requested
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(f'{output_dir}/{name}.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir}/{name}.png")
    
    return figures


if __name__ == "__main__":
    print("ComprehensiveEvaluator module loaded successfully")
