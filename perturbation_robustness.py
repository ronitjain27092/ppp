"""
PERTURBATION & ROBUSTNESS TESTING MODULE
========================================

Tests model generalization by adding noise and perturbations to data.

WHY THIS MATTERS FOR LEAK DETECTION:
- If model accuracy drops to 0% with tiny noise, data leakage is confirmed
- Honest models gracefully degrade with realistic perturbations
- Perfect models that collapse with noise are clearly overfitting

CONCEPTS:
1. Gaussian Noise: Add random noise to features
2. Feature Shuffling: Randomly shuffle feature values
3. Feature Dropout: Random feature masking
4. Adversarial Perturbation: Systematic perturbation toward misclassification
5. Cross-validation with perturbation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


class PerturbationRobustness:
    """Test model robustness to data perturbations."""
    
    def __init__(self, model_predict_func, model_predict_proba_func=None):
        """
        Initialize with model prediction functions.
        
        Args:
            model_predict_func: function(X) -> y_pred (binary predictions)
            model_predict_proba_func: function(X) -> y_proba (probabilities)
        """
        self.predict = model_predict_func
        self.predict_proba = model_predict_proba_func
        self.results = {}
    
    # =====================================================================
    # 1. GAUSSIAN NOISE ROBUSTNESS
    # =====================================================================
    def test_gaussian_noise(self, X_test, y_test, noise_levels=None):
        """
        Test accuracy with Gaussian noise at increasing levels.
        
        For leakage: Accuracy drops dramatically with tiny noise
        For honest model: Gradual degradation
        """
        if noise_levels is None:
            noise_levels = np.arange(0, 1.1, 0.1)  # 0% to 100% of std
        
        results = {'noise_level': [], 'accuracy': [], 'f1': []}
        
        X_test = X_test.values if hasattr(X_test, 'values') else X_test.astype(float)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        # Baseline accuracy (no noise)
        y_pred_clean = self.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_pred_clean)
        baseline_f1 = f1_score(y_test, y_pred_clean, zero_division=0)
        
        results['noise_level'].append(0.0)
        results['accuracy'].append(baseline_acc)
        results['f1'].append(baseline_f1)
        
        print("\n" + "="*70)
        print("GAUSSIAN NOISE ROBUSTNESS TEST")
        print("="*70)
        print(f"Baseline (no noise): Accuracy={baseline_acc:.4f}, F1={baseline_f1:.4f}\n")
        
        X_std = np.std(X_test, axis=0)
        
        for noise_level in noise_levels[1:]:  # Skip 0.0 (already done)
            # Add Gaussian noise scaled by feature std
            X_noisy = X_test + np.random.normal(0, noise_level * X_std, X_test.shape)
            
            y_pred_noisy = self.predict(X_noisy)
            acc = accuracy_score(y_test, y_pred_noisy)
            f1 = f1_score(y_test, y_pred_noisy, zero_division=0)
            
            results['noise_level'].append(noise_level)
            results['accuracy'].append(acc)
            results['f1'].append(f1)
            
            acc_drop = (baseline_acc - acc) * 100
            print(f"Noise={noise_level:>5.1%}: Accuracy={acc:.4f} (Δ={-acc_drop:>6.1f}%), "
                  f"F1={f1:.4f}")
        
        print("="*70)
        
        # Interpretation
        min_acc = min(results['accuracy'][1:])
        if min_acc < 0.6:
            print(f"\n⚠️  WARNING: Accuracy drops below 60% with moderate noise")
            print(f"   This suggests model is vulnerable to perturbations (leakage indicator)")
        else:
            print(f"\n✓ Model maintains reasonable accuracy with noise (robustness indicator)")
        
        self.results['gaussian_noise'] = results
        return results
    
    # =====================================================================
    # 2. FEATURE SHUFFLING
    # =====================================================================
    def test_feature_shuffling(self, X_test, y_test, shuffle_fraction=0.3):
        """
        Randomly shuffle feature values to test significance.
        
        For leakage: Model robust to shuffling (makes little difference)
        For honest model: Shuffling important features hurts performance
        """
        X_test = X_test.values if hasattr(X_test, 'values') else X_test.astype(float)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        n_features = X_test.shape[1]
        n_shuffle = int(n_features * shuffle_fraction)
        
        # Baseline
        y_pred_clean = self.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_pred_clean)
        baseline_f1 = f1_score(y_test, y_pred_clean, zero_division=0)
        
        print("\n" + "="*70)
        print(f"FEATURE SHUFFLING TEST (shuffle {shuffle_fraction*100:.0f}% of {n_features} features)")
        print("="*70)
        print(f"Baseline: Accuracy={baseline_acc:.4f}, F1={baseline_f1:.4f}\n")
        
        results = {'features_shuffled': n_shuffle, 'accuracy_drop': []}
        
        # Test with random feature shuffling
        accs = []
        for trial in range(10):  # 10 random shuffles
            X_shuffled = X_test.copy()
            feat_indices = np.random.choice(n_features, n_shuffle, replace=False)
            
            for feat_idx in feat_indices:
                np.random.shuffle(X_shuffled[:, feat_idx])
            
            y_pred_shuffled = self.predict(X_shuffled)
            acc = accuracy_score(y_test, y_pred_shuffled)
            accs.append(acc)
            acc_drop = (baseline_acc - acc) * 100
            results['accuracy_drop'].append(acc_drop)
        
        mean_drop = np.mean(results['accuracy_drop'])
        std_drop = np.std(results['accuracy_drop'])
        
        print(f"After shuffling: Accuracy={np.mean(accs):.4f} "
              f"(Δ={-mean_drop:>6.1f}% ± {std_drop:.1f}%)")
        print("="*70)
        
        if mean_drop < 2:
            print(f"\n✓ Model is robust to shuffling (features may not matter much)")
        else:
            print(f"\n✓ Model degrades with shuffling (uses features properly)")
        
        self.results['feature_shuffling'] = results
        return results
    
    # =====================================================================
    # 3. FEATURE DROPOUT (Random Masking)
    # =====================================================================
    def test_feature_dropout(self, X_test, y_test, dropout_fractions=None):
        """
        Randomly mask out (zero) percentage of features.
        
        For leakage: Model still predicts well (only one/two features matter)
        For honest model: Drops significantly with too much dropout
        """
        if dropout_fractions is None:
            dropout_fractions = np.linspace(0, 0.8, 9)
        
        X_test = X_test.values if hasattr(X_test, 'values') else X_test.astype(float)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        # Baseline
        y_pred_clean = self.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_pred_clean)
        baseline_f1 = f1_score(y_test, y_pred_clean, zero_division=0)
        
        results = {'dropout_fraction': [], 'accuracy': [], 'f1': []}
        
        print("\n" + "="*70)
        print("FEATURE DROPOUT TEST (mask random features)")
        print("="*70)
        print(f"Baseline: Accuracy={baseline_acc:.4f}, F1={baseline_f1:.4f}\n")
        print(f"{'Dropout':<10} {'Accuracy':<12} {'F1':<12} {'Δ Accuracy':<12}")
        print("-"*70)
        
        X_test_float = X_test.astype(float)
        
        for dropout_frac in dropout_fractions:
            X_dropped = X_test_float.copy()
            n_features = X_dropped.shape[1]
            n_dropout = int(n_features * dropout_frac)
            
            # Randomly select features to dropout
            dropout_indices = np.random.choice(n_features, n_dropout, replace=False)
            X_dropped[:, dropout_indices] = 0
            
            y_pred_dropped = self.predict(X_dropped)
            acc = accuracy_score(y_test, y_pred_dropped)
            f1 = f1_score(y_test, y_pred_dropped, zero_division=0)
            
            results['dropout_fraction'].append(dropout_frac)
            results['accuracy'].append(acc)
            results['f1'].append(f1)
            
            acc_drop = (baseline_acc - acc) * 100
            print(f"{dropout_frac:>6.0%}      {acc:>10.4f}  {f1:>10.4f}  {-acc_drop:>10.1f}%")
        
        print("="*70)
        
        # At 50% dropout, realistic model should lose 10-30% accuracy
        acc_at_50 = results['accuracy'][int(len(results['accuracy'])/2)]
        drop_at_50 = (baseline_acc - acc_at_50) * 100
        
        if drop_at_50 < 5:
            print(f"\n⚠️  Model retains {acc_at_50:.1%} accuracy with 50% features masked")
            print(f"   This suggests only few features matter (LEAKAGE INDICATOR)")
        elif drop_at_50 > 50:
            print(f"\n⚠️  Model completely fails with 50% dropout")
            print(f"   This suggests over-reliance on specific features")
        else:
            print(f"\n✓ Model shows reasonable degradation with feature dropout")
        
        self.results['feature_dropout'] = results
        return results
    
    # =====================================================================
    # 4. ADVERSARIAL NOISE
    # =====================================================================
    def test_adversarial_noise(self, X_test, y_test, n_iterations=10):
        """
        Add noise in direction that increases misclassification.
        
        For leakage: Tiny perturbations cause many misclassifications
        For honest model: Needs larger perturbations to fool
        """
        X_test = X_test.values if hasattr(X_test, 'values') else X_test.astype(float)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_test = y_test.flatten()
        
        if self.predict_proba is None:
            print("⚠️  Adversarial test requires predict_proba function")
            return None
        
        print("\n" + "="*70)
        print("ADVERSARIAL PERTURBATION TEST")
        print("="*70)
        
        results = {'perturbation_magnitude': [], 'accuracy': []}
        
        # Baseline
        y_proba = self.predict_proba(X_test)
        y_proba = y_proba.flatten()
        y_pred_clean = (y_proba > 0.5).astype(int)
        baseline_acc = accuracy_score(y_test, y_pred_clean)
        
        print(f"Baseline accuracy: {baseline_acc:.4f}\n")
        
        perturbation_scales = np.logspace(-3, -0.5, n_iterations)  # 0.001 to 0.316
        
        for scale in perturbation_scales:
            X_perturbed = X_test.copy()
            
            # Simple adversarial: push toward wrong class
            y_proba_adv = self.predict_proba(X_perturbed)
            y_proba_adv = y_proba_adv.flatten()
            
            # Perturbation direction: toward opposite class
            for i in range(len(X_perturbed)):
                target = 1 - y_test[i]  # Opposite class
                gradient_direction = np.random.randn(X_perturbed.shape[1])
                gradient_direction /= np.linalg.norm(gradient_direction)
                X_perturbed[i] += scale * gradient_direction
            
            y_pred_perturbed = self.predict(X_perturbed)
            acc = accuracy_score(y_test, y_pred_perturbed)
            
            results['perturbation_magnitude'].append(scale)
            results['accuracy'].append(acc)
            
            acc_drop = (baseline_acc - acc) * 100
            print(f"Scale={scale:.4f}: Accuracy={acc:.4f} (Δ={-acc_drop:>6.1f}%)")
        
        print("="*70)
        
        self.results['adversarial_noise'] = results
        return results
    
    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    def plot_all_perturbation_results(self):
        """Plot all perturbation test results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        
        # 1. Gaussian Noise
        if 'gaussian_noise' in self.results:
            ax = axes[0, 0]
            data = self.results['gaussian_noise']
            ax.plot(data['noise_level'], data['accuracy'], 'o-', linewidth=2.5, markersize=8, color='tab:blue')
            ax.set_xlabel('Noise Level (fraction of std)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title('Gaussian Noise Robustness', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='60% threshold')
            ax.legend()
        
        # 2. Feature Dropout
        if 'feature_dropout' in self.results:
            ax = axes[0, 1]
            data = self.results['feature_dropout']
            ax.plot(data['dropout_fraction'], data['accuracy'], 'o-', linewidth=2.5, markersize=8, color='tab:orange')
            ax.set_xlabel('Feature Dropout Fraction', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title('Feature Dropout Robustness', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            ax.legend()
        
        # 3. Feature Shuffling
        if 'feature_shuffling' in self.results:
            ax = axes[1, 0]
            data = self.results['feature_shuffling']
            drops = data['accuracy_drop']
            ax.bar(['Shuffled\nFeatures'], [np.mean(drops)], yerr=[np.std(drops)],
                  capsize=5, color='tab:green', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Accuracy Drop (%)', fontsize=11, fontweight='bold')
            ax.set_title('Feature Shuffling: Robustness', fontsize=12, fontweight='bold')
            ax.set_ylim([0, max(drops) + 10])
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Adversarial Noise
        if 'adversarial_noise' in self.results:
            ax = axes[1, 1]
            data = self.results['adversarial_noise']
            ax.plot(data['perturbation_magnitude'], data['accuracy'], 'o-', linewidth=2.5, markersize=8, color='tab:red')
            ax.set_xlabel('Perturbation Scale', fontsize=11, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            ax.set_title('Adversarial Noise Robustness', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim([0, 1.05])
        
        plt.suptitle('Model Robustness to Perturbations\n'
                    '(Steep drops indicate leakage; gradual degradation indicates honest model)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def run_full_perturbation_suite(model_predict, X_test, y_test,
                               model_predict_proba=None, output_dir=None):
    """
    Run all perturbation tests and generate report.
    
    Args:
        model_predict: function(X) -> y_pred
        X_test: Test features
        y_test: Test labels
        model_predict_proba: function(X) -> y_proba (optional)
        output_dir: Directory to save plots
    """
    tester = PerturbationRobustness(model_predict, model_predict_proba)
    
    print("\n" + "="*80)
    print("RUNNING FULL PERTURBATION ROBUSTNESS SUITE")
    print("="*80)
    
    tester.test_gaussian_noise(X_test, y_test)
    tester.test_feature_shuffling(X_test, y_test)
    tester.test_feature_dropout(X_test, y_test)
    
    if model_predict_proba:
        tester.test_adversarial_noise(X_test, y_test)
    
    fig = tester.plot_all_perturbation_results()
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f'{output_dir}/perturbation_robustness.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_dir}/perturbation_robustness.png")
    
    return tester


if __name__ == "__main__":
    print("PerturbationRobustness module loaded successfully")
