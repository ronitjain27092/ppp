"""
FEATURE IMPORTANCE & LEAKAGE DETECTION MODULE
==============================================

Identifies features that are artificially predictive (leakage indicators).

THE PROBLEM:
If a single feature has very high importance and is strongly correlated with the label,
it likely encodes the label directly (data leakage), which causes perfect accuracy.

This module provides multiple perspectives on feature importance to detect leakage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


class FeatureLeakageDetector:
    """Detect feature leakage through importance analysis."""
    
    def __init__(self, X_train, y_train, feature_names=None):
        """
        Initialize detector.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Optional feature names
        """
        self.X_train = X_train.values if hasattr(X_train, 'values') else X_train.astype(float)
        self.y_train = y_train.values if hasattr(y_train, 'values') else y_train.flatten()
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]
        self.feature_names = feature_names
        
        self.importance_dict = {}
        self.leakage_indicators = []
    
    # =====================================================================
    # 1. CORRELATION ANALYSIS
    # =====================================================================
    def analyze_correlations(self, correlation_threshold=0.8):
        """
        Analyze feature-label correlations (Pearson and Spearman).
        
        High correlation = potential feature leakage
        
        Args:
            correlation_threshold: Flag features with |correlation| > threshold
        """
        print("\n" + "="*70)
        print("1. CORRELATION ANALYSIS (Pearson & Spearman)")
        print("="*70)
        
        correlations = {}
        
        for i, feat_name in enumerate(self.feature_names):
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(self.X_train[:, i], self.y_train)
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(self.X_train[:, i], self.y_train)
            
            correlations[feat_name] = {
                'pearson_r': abs(pearson_r),
                'spearman_r': abs(spearman_r),
                'pearson_p': pearson_p,
                'spearman_p': spearman_p,
                'max_correlation': max(abs(pearson_r), abs(spearman_r))
            }
        
        # Sort by max correlation
        sorted_corrs = sorted(correlations.items(),
                             key=lambda x: x[1]['max_correlation'],
                             reverse=True)
        
        print(f"\nTop 10 features by correlation with label:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Feature':<25} {'Pearson':<12} {'Spearman':<12}")
        print("-" * 70)
        
        leaky_features = []
        for rank, (feat_name, corr_dict) in enumerate(sorted_corrs[:10], 1):
            pearson = corr_dict['pearson_r']
            spearman = corr_dict['spearman_r']
            
            marker = ""
            if max(pearson, spearman) > correlation_threshold:
                marker = " 🔴 LEAKY"
                leaky_features.append((feat_name, max(pearson, spearman)))
            
            print(f"{rank:<5} {feat_name:<25} {pearson:>11.4f} {spearman:>11.4f}{marker}")
        
        if leaky_features:
            print(f"\n🔴 {len(leaky_features)} FEATURES with correlation > {correlation_threshold}")
            for feat, corr in leaky_features:
                self.leakage_indicators.append({
                    'type': 'High Correlation',
                    'feature': feat,
                    'value': float(corr),
                    'severity': 'CRITICAL' if corr > 0.95 else 'HIGH'
                })
        else:
            print(f"\n✓ No features with correlation > {correlation_threshold}")
        
        self.importance_dict['correlations'] = dict(sorted_corrs)
        return dict(sorted_corrs)
    
    # =====================================================================
    # 2. MUTUAL INFORMATION
    # =====================================================================
    def analyze_mutual_information(self):
        """
        Analyze mutual information (MI) between features and label.
        
        MI captures non-linear relationships.
        High MI = feature predicts label well = potential leakage
        """
        print("\n" + "="*70)
        print("2. MUTUAL INFORMATION ANALYSIS")
        print("="*70)
        
        mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
        
        # Normalize MI scores
        mi_scores_norm = mi_scores / (mi_scores.max() + 1e-10)
        
        mi_dict = {
            self.feature_names[i]: {
                'raw_mi': float(mi_scores[i]),
                'normalized_mi': float(mi_scores_norm[i])
            }
            for i in range(len(self.feature_names))
        }
        
        sorted_mi = sorted(mi_dict.items(),
                          key=lambda x: x[1]['raw_mi'],
                          reverse=True)
        
        print(f"\nTop 10 features by Mutual Information:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Feature':<25} {'Raw MI':<12} {'Normalized':<12}")
        print("-" * 70)
        
        suspicious_mi = []
        for rank, (feat_name, mi_vals) in enumerate(sorted_mi[:10], 1):
            raw_mi = mi_vals['raw_mi']
            norm_mi = mi_vals['normalized_mi']
            
            marker = ""
            if norm_mi > 0.7:
                marker = " 🔴 SUSPICIOUS"
                suspicious_mi.append((feat_name, norm_mi))
            
            print(f"{rank:<5} {feat_name:<25} {raw_mi:>11.4f} {norm_mi:>11.4f}{marker}")
        
        if suspicious_mi:
            print(f"\n🔴 {len(suspicious_mi)} FEATURES with normalized MI > 0.7")
            for feat, mi in suspicious_mi:
                if not any(ind['feature'] == feat for ind in self.leakage_indicators):
                    self.leakage_indicators.append({
                        'type': 'High Mutual Information',
                        'feature': feat,
                        'value': float(mi),
                        'severity': 'HIGH'
                    })
        else:
            print(f"\n✓ No features with normalized MI > 0.7")
        
        self.importance_dict['mutual_information'] = dict(sorted_mi)
        return dict(sorted_mi)
    
    # =====================================================================
    # 3. RANDOM FOREST FEATURE IMPORTANCE
    # =====================================================================
    def analyze_random_forest_importance(self, n_estimators=100):
        """
        Train Random Forest to get feature importances.
        
        Warning: If one feature gets 50%+ importance, it's suspicious.
        """
        print("\n" + "="*70)
        print("3. RANDOM FOREST FEATURE IMPORTANCE")
        print("="*70)
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        
        importances = {
            self.feature_names[i]: float(rf.feature_importances_[i])
            for i in range(len(self.feature_names))
        }
        
        sorted_imp = sorted(importances.items(),
                           key=lambda x: x[1],
                           reverse=True)
        
        print(f"\nTop 10 features by RF Importance:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Feature':<25} {'Importance':<12} {'Percentage':<12}")
        print("-" * 70)
        
        total_imp = sum(importances.values())
        suspicious_rf = []
        
        for rank, (feat_name, imp) in enumerate(sorted_imp[:10], 1):
            pct = (imp / total_imp) * 100 if total_imp > 0 else 0
            
            marker = ""
            if pct > 30:
                marker = " 🔴 DOMINANT"
                suspicious_rf.append((feat_name, pct))
            elif pct > 20:
                marker = " 🟡 NOTABLE"
            
            print(f"{rank:<5} {feat_name:<25} {imp:>11.4f} {pct:>11.1f}%{marker}")
        
        if suspicious_rf:
            print(f"\n🟡 WARNING: Top feature(s) with >30% importance:")
            for feat, pct in suspicious_rf:
                if not any(ind['feature'] == feat for ind in self.leakage_indicators):
                    self.leakage_indicators.append({
                        'type': 'Feature Dominance',
                        'feature': feat,
                        'value': float(pct),
                        'severity': 'HIGH'
                    })
        else:
            print(f"\n✓ Feature importances reasonably distributed")
        
        self.importance_dict['random_forest'] = dict(sorted_imp)
        return dict(sorted_imp)
    
    # =====================================================================
    # 4. PERMUTATION IMPORTANCE
    # =====================================================================
    def analyze_permutation_importance(self, X_test, y_test, model, n_repeats=10):
        """
        Permutation importance: shuffle feature and see accuracy drop.
        
        High drop when shuffled = important feature
        
        Args:
            X_test: Test features
            y_test: Test labels
            model: Fitted model with predict() method
            n_repeats: Number of shuffles
        """
        print("\n" + "="*70)
        print("4. PERMUTATION IMPORTANCE (on Test Set)")
        print("="*70)
        
        X_test = X_test.values if hasattr(X_test, 'values') else X_test.astype(float)
        
        perm_imp = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        perm_dict = {
            self.feature_names[i]: {
                'importance': float(perm_imp.importances_mean[i]),
                'std': float(perm_imp.importances_std[i])
            }
            for i in range(len(self.feature_names))
        }
        
        sorted_perm = sorted(perm_dict.items(),
                            key=lambda x: x[1]['importance'],
                            reverse=True)
        
        print(f"\nTop 10 features by Permutation Importance:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Feature':<25} {'Importance':<15} {'Std':<12}")
        print("-" * 70)
        
        for rank, (feat_name, imp_dict) in enumerate(sorted_perm[:10], 1):
            imp = imp_dict['importance']
            std = imp_dict['std']
            
            marker = ""
            if imp > 0.1:
                marker = " 🔴"
            
            print(f"{rank:<5} {feat_name:<25} {imp:>14.4f} {std:>11.4f}{marker}")
        
        print(f"\n✓ Permutation importance computed on test set (no data leakage)")
        
        self.importance_dict['permutation'] = dict(sorted_perm)
        return dict(sorted_perm)
    
    # =====================================================================
    # 5. FEATURE VARIANCE ANALYSIS
    # =====================================================================
    def analyze_feature_variance(self):
        """
        Check if some features have extremely low variance.
        
        Low variance + high importance = potential constant/leakage feature
        """
        print("\n" + "="*70)
        print("5. FEATURE VARIANCE ANALYSIS")
        print("="*70)
        
        variances = np.var(self.X_train, axis=0)
        
        var_dict = {
            self.feature_names[i]: {
                'variance': float(variances[i]),
                'std': float(np.std(self.X_train[:, i]))
            }
            for i in range(len(self.feature_names))
        }
        
        sorted_var = sorted(var_dict.items(),
                           key=lambda x: x[1]['variance'],
                           reverse=True)
        
        print(f"\nFeature Variance (Top and Bottom 5):")
        print("-" * 70)
        print(f"{'Rank':<5} {'Feature':<25} {'Variance':<15} {'Std Dev':<12}")
        print("-" * 70)
        
        # Top 5
        print("Highest variance:")
        for rank, (feat_name, var_dict_item) in enumerate(sorted_var[:5], 1):
            var = var_dict_item['variance']
            std = var_dict_item['std']
            print(f"{rank:<5} {feat_name:<25} {var:>14.6f} {std:>11.4f}")
        
        # Bottom 5
        print("\nLowest variance (potential constant features):")
        for rank, (feat_name, var_dict_item) in enumerate(sorted_var[-5:], 1):
            var = var_dict_item['variance']
            std = var_dict_item['std']
            marker = " 🟡 LOW VAR" if var < 0.01 else ""
            print(f"{rank:<5} {feat_name:<25} {var:>14.6f} {std:>11.4f}{marker}")
        
        low_var_features = [f for f, v in sorted_var if v[1]['variance'] < 0.001]
        if low_var_features:
            print(f"\n🟡 {len(low_var_features)} near-constant features detected:")
            for feat in low_var_features:
                print(f"   - {feat}")
        else:
            print(f"\n✓ No near-constant features detected")
        
        self.importance_dict['variance'] = dict(sorted_var)
        return dict(sorted_var)
    
    # =====================================================================
    # 6. COMBINATIONS - SUSPICIOUS PATTERNS
    # =====================================================================
    def detect_leakage_patterns(self):
        """
        Identify suspicious patterns that indicate leakage.
        """
        print("\n" + "="*70)
        print("LEAKAGE PATTERN DETECTION")
        print("="*70)
        
        if not self.leakage_indicators:
            print("\n✅ No obvious leakage patterns detected")
            return []
        
        print(f"\n🔴 {len(self.leakage_indicators)} Leakage Indicators Found:")
        print("-" * 70)
        
        for i, indicator in enumerate(self.leakage_indicators, 1):
            print(f"\n{i}. {indicator['type']}")
            print(f"   Feature: {indicator['feature']}")
            print(f"   Value: {indicator['value']:.4f}")
            print(f"   Severity: {indicator['severity']} 🔴" if indicator['severity'] == 'CRITICAL' else f"   Severity: {indicator['severity']} 🟡")
        
        return self.leakage_indicators
    
    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    def plot_importance_comparison(self):
        """
        Plot all importance measures side-by-side.
        """
        if not self.importance_dict:
            print("⚠️  No importance data to plot. Run analyze_* methods first.")
            return None
        
        # Select top features across all methods
        all_features = set()
        for imp_dict in self.importance_dict.values():
            if isinstance(imp_dict, dict):
                all_features.update(imp_dict.keys())
        
        all_features = sorted(all_features)[:15]  # Top 15
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Correlations
        if 'correlations' in self.importance_dict:
            ax = axes[0, 0]
            data = self.importance_dict['correlations']
            features = [f for f, _ in sorted(data.items(), key=lambda x: x[1]['max_correlation'], reverse=True)[:10]]
            values = [data[f]['max_correlation'] for f in features]
            
            colors = ['red' if v > 0.8 else 'orange' if v > 0.6 else 'blue' for v in values]
            ax.barh(features, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Max |Correlation|', fontsize=11, fontweight='bold')
            ax.set_title('Correlation with Label', fontsize=12, fontweight='bold')
            ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Leakage threshold')
            ax.legend()
            ax.set_xlim([0, 1])
        
        # 2. Mutual Information
        if 'mutual_information' in self.importance_dict:
            ax = axes[0, 1]
            data = self.importance_dict['mutual_information']
            features = [f for f, _ in sorted(data.items(), key=lambda x: x[1]['normalized_mi'], reverse=True)[:10]]
            values = [data[f]['normalized_mi'] for f in features]
            
            colors = ['red' if v > 0.7 else 'orange' if v > 0.5 else 'blue' for v in values]
            ax.barh(features, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Normalized MI', fontsize=11, fontweight='bold')
            ax.set_title('Mutual Information', fontsize=12, fontweight='bold')
            ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5)
            ax.set_xlim([0, 1])
        
        # 3. Random Forest Importance
        if 'random_forest' in self.importance_dict:
            ax = axes[1, 0]
            data = self.importance_dict['random_forest']
            features = [f for f, _ in sorted(data.items(), key=lambda x: x[1], reverse=True)[:10]]
            values = [data[f] for f in features]
            total = sum(self.importance_dict['random_forest'].values())
            values_pct = [v / total * 100 for v in values]
            
            colors = ['red' if v > 30 else 'orange' if v > 20 else 'blue' for v in values_pct]
            ax.barh(features, values_pct, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
            ax.set_title('Random Forest Importance', fontsize=12, fontweight='bold')
            ax.axvline(x=30, color='red', linestyle='--', alpha=0.5)
        
        # 4. Feature Variance
        if 'variance' in self.importance_dict:
            ax = axes[1, 1]
            data = self.importance_dict['variance']
            features = [f for f, _ in sorted(data.items(), key=lambda x: np.log(x[1]['variance']+1e-10), reverse=True)[:10]]
            values = [np.log(data[f]['variance']+1e-10) for f in features]
            
            ax.barh(features, values, color='blue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Log(Variance)', fontsize=11, fontweight='bold')
            ax.set_title('Feature Variance', fontsize=12, fontweight='bold')
        
        plt.suptitle('Feature Importance Comparison\n(Red = Suspicious/Leaky, Yellow = Notable, Blue = Normal)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig


# =====================================================================
# COMPREHENSIVE FEATURE LEAKAGE REPORT
# =====================================================================
def generate_feature_leakage_report(X_train, y_train, X_test, y_test,
                                   model, feature_names=None, output_dir=None):
    """
    Generate comprehensive feature leakage report.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model: Fitted model with predict() method
        feature_names: Optional feature names
        output_dir: Optional directory to save plots
    """
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE FEATURE LEAKAGE ANALYSIS")
    print("="*80)
    
    detector = FeatureLeakageDetector(X_train, y_train, feature_names)
    
    # Run all analyses
    detector.analyze_correlations(correlation_threshold=0.8)
    detector.analyze_mutual_information()
    detector.analyze_random_forest_importance()
    detector.analyze_feature_variance()
    
    # Permutation importance on test set
    try:
        detector.analyze_permutation_importance(X_test, y_test, model)
    except Exception as e:
        print(f"\n⚠️  Could not compute permutation importance: {e}")
    
    # Detect patterns
    leakage_patterns = detector.detect_leakage_patterns()
    
    # Visualize
    fig = detector.plot_importance_comparison()
    
    if output_dir and fig:
        import os
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f'{output_dir}/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {output_dir}/feature_importance_analysis.png")
    
    return detector, leakage_patterns


if __name__ == "__main__":
    print("FeatureLeakageDetector module loaded successfully")
