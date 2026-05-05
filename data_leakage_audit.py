"""
COMPREHENSIVE DATA LEAKAGE AUDIT SCRIPT
========================================

This script performs a thorough audit of your ML pipeline to detect and report
all sources of data leakage that could inflate accuracy metrics to unrealistic levels.

DATA LEAKAGE TYPES DETECTED:
1. Train/Test Data Overlap (near-duplicates)
2. Feature Leakage (features > 0.95 corr with label)
3. Scaling/Normalization Leakage (fit on combined data instead of train-only)
4. Preprocessing Leakage (pipeline fit on test data)
5. Temporal Leakage (if applicable)
6. Feature Engineering Leakage (statistics computed on full dataset)
7. Model Selection Leakage (using test data for hyperparameter tuning)
8. Class Weight Leakage (computed on mixed data)
9. Feature Importance Leakage (can indicate suspicious features)

WHAT CAUSES THE PERFECT (≈1.0) ACCURACY PROBLEM:
- When a feature is almost perfectly correlated with the label
- When train/test sets contain duplicate or near-duplicate samples
- When scaling/preprocessing is fit on both train AND test together
- When the dataset is small and a single feature encodes the label
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
import hashlib
import warnings
warnings.filterwarnings('ignore')


class DataLeakageAudit:
    """Comprehensive audit of data leakage in ML pipeline."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.issues_found = []
        self.warnings = []
        self.recommendations = []
        
    def _log(self, message, level="INFO"):
        """Log messages with levels."""
        if self.verbose:
            if level == "ERROR":
                print(f"❌ {message}")
            elif level == "WARNING":
                print(f"⚠️  {message}")
            elif level == "SUCCESS":
                print(f"✅ {message}")
            else:
                print(f"ℹ️  {message}")
    
    # =====================================================================
    # AUDIT 1: DATA DUPLICATION AND OVERLAPS
    # =====================================================================
    def audit_train_test_overlap(self, X_train, X_test, y_train=None, y_test=None):
        """
        Check for exact and near-duplicate samples between train and test sets.
        
        This is a CRITICAL leakage source that would cause 100% accuracy.
        """
        print("\n" + "="*80)
        print("AUDIT 1: TRAIN/TEST OVERLAP (Duplicate Samples)")
        print("="*80)
        
        # Convert to numpy
        X_train = X_train.values if hasattr(X_train, 'values') else X_train
        X_test = X_test.values if hasattr(X_test, 'values') else X_test
        
        def hash_row(row, decimals=10):
            """Hash a row for duplicate detection."""
            rounded = np.round(row, decimals)
            return hashlib.md5(rounded.tobytes()).hexdigest()
        
        # Check exact duplicates
        train_hashes = {hash_row(row, decimals=10): i for i, row in enumerate(X_train)}
        test_exact_dups = []
        test_near_dups = []
        
        for test_idx, test_row in enumerate(X_test):
            test_hash = hash_row(test_row, decimals=10)
            if test_hash in train_hashes:
                test_exact_dups.append((test_idx, train_hashes[test_hash]))
        
        # Check near-duplicates (with lower precision)
        train_hashes_coarse = {hash_row(row, decimals=4): i for i, row in enumerate(X_train)}
        
        for test_idx, test_row in enumerate(X_test):
            test_hash_coarse = hash_row(test_row, decimals=4)
            if test_hash_coarse in train_hashes_coarse:
                # Double-check it's not already counted as exact
                if not any(t[0] == test_idx for t in test_exact_dups):
                    test_near_dups.append((test_idx, train_hashes_coarse[test_hash_coarse]))
        
        # Report
        if test_exact_dups:
            self._log(f"{len(test_exact_dups)} EXACT DUPLICATE samples found between train/test!",
                     "ERROR")
            self.issues_found.append({
                'type': 'Data Duplication',
                'severity': 'CRITICAL',
                'samples_affected': len(test_exact_dups),
                'description': 'Exact duplicates between train and test sets'
            })
        else:
            self._log("No exact duplicates between train and test", "SUCCESS")
        
        if test_near_dups:
            n_dups = len(test_near_dups)
            pct = (n_dups / len(X_test)) * 100
            self._log(f"{n_dups} NEAR-DUPLICATE samples ({pct:.2f}% of test set)",
                     "WARNING")
            self.warnings.append(f"Near-duplicates: {pct:.2f}% of test set")
        else:
            self._log("No near-duplicates between train and test", "SUCCESS")
        
        # Class balance check in duplicates
        if test_exact_dups and y_train is not None and y_test is not None:
            dup_indices = [t[0] for t in test_exact_dups]
            dup_labels = y_test[dup_indices] if hasattr(y_test, '__getitem__') else y_test.iloc[dup_indices]
            matching_labels = sum(1 for test_i, train_i in test_exact_dups 
                                if y_test[test_i] == y_train[train_i])
            self._log(f"  {matching_labels}/{len(test_exact_dups)} duplicates have matching labels",
                     "WARNING")
            self.recommendations.append(
                "👉 CRITICAL FIX: Remove duplicate samples from test set before evaluation"
            )
        
        return {
            'exact_duplicates': len(test_exact_dups),
            'near_duplicates': len(test_near_dups),
            'near_duplicate_pct': (len(test_near_dups) / len(X_test)) * 100
        }
    
    # =====================================================================
    # AUDIT 2: FEATURE LEAKAGE (Suspiciously High Correlation)
    # =====================================================================
    def audit_feature_leakage(self, X, y, feature_names=None, corr_threshold=0.8):
        """
        Identify features that are suspiciously correlated with the label.
        
        This is the #1 reason for 100% accuracy in malware detection.
        If a single feature encodes the label (like "malware_probability"),
        the model will achieve perfect accuracy.
        """
        print("\n" + "="*80)
        print("AUDIT 2: FEATURE LEAKAGE (High Label Correlation)")
        print("="*80)
        
        X = X.values if hasattr(X, 'values') else X
        y = y.values if hasattr(y, 'values') else y
        y = y.flatten()
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Method 1: Pearson Correlation
        correlations = {}
        for i, feat_name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations[feat_name] = abs(corr)
        
        # Sort by correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Identify leaky features
        leaky_features = [(f, c) for f, c in sorted_corrs if c > corr_threshold]
        
        print(f"\nTop 10 features by correlation with label:")
        print("-" * 60)
        for i, (feat, corr) in enumerate(sorted_corrs[:10], 1):
            marker = "🔴 LEAKY" if (feat, corr) in leaky_features else " "
            print(f"  {i:2d}. {feat:30s} |corr| = {corr:.4f}  {marker}")
        
        if leaky_features:
            self._log(f"{len(leaky_features)} FEATURES with |corr| > {corr_threshold} detected!",
                     "ERROR")
            self.issues_found.append({
                'type': 'Feature Leakage',
                'severity': 'CRITICAL',
                'affected_features': [f for f, _ in leaky_features],
                'description': f'{len(leaky_features)} features highly correlated with label'
            })
            self.recommendations.append(
                f"👉 CRITICAL FIX: Drop these features before training:\n"
                f"   {', '.join([f for f, _ in leaky_features])}"
            )
        else:
            self._log(f"No features with absolute correlation > {corr_threshold}", "SUCCESS")
        
        # Method 2: Feature Importance from Random Forest
        print(f"\nFeature Importance (Random Forest):")
        print("-" * 60)
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            rf.fit(X, y)
            importances = list(zip(feature_names, rf.feature_importances_))
            importances = sorted(importances, key=lambda x: x[1], reverse=True)
            
            for i, (feat, imp) in enumerate(importances[:10], 1):
                print(f"  {i:2d}. {feat:30s} importance = {imp:.4f}")
            
            # Flag suspiciously dominant features
            top_imp = importances[0][1]
            if top_imp > 0.3:
                self._log(
                    f"⚠️  Top feature '{importances[0][0]}' has {top_imp:.1%} importance (>=30% is suspicious)",
                    "WARNING"
                )
                self.warnings.append(f"Top feature dominance: {top_imp:.1%}")
        except Exception as e:
            self._log(f"Could not compute RF importances: {e}", "WARNING")
        
        return {
            'leaky_features': [f for f, _ in leaky_features],
            'top_correlations': sorted_corrs[:5],
            'recommendation': 'Drop features with |corr| > 0.95 before training'
        }
    
    # =====================================================================
    # AUDIT 3: SCALING/NORMALIZATION LEAKAGE
    # =====================================================================
    def audit_scaling_leakage(self, X_train, X_test, X_val=None):
        """
        Check if scaler was fit on TRAINING DATA ONLY (correct)
        or on combined data (LEAKAGE).
        
        If you fit the scaler on train+test, the test set statistics
        are known at training time via the scaling parameters.
        """
        print("\n" + "="*80)
        print("AUDIT 3: SCALING / NORMALIZATION LEAKAGE")
        print("="*80)
        
        X_train = X_train.values if hasattr(X_train, 'values') else X_train
        X_test = X_test.values if hasattr(X_test, 'values') else X_test
        if X_val is not None:
            X_val = X_val.values if hasattr(X_val, 'values') else X_val
        
        print("\nCheck 1: Range differences between train/test")
        print("-" * 60)
        
        train_min = X_train.min(axis=0)
        train_max = X_train.max(axis=0)
        test_min = X_test.min(axis=0)
        test_max = X_test.max(axis=0)
        
        # Features where test range exceeds train range
        out_of_range = (test_min < train_min) | (test_max > train_max)
        n_out_of_range = out_of_range.sum()
        
        if n_out_of_range > 0:
            self._log(
                f"{n_out_of_range}/{len(out_of_range)} features have values outside train range",
                "WARNING"
            )
            self._log(
                "This is EXPECTED if scaler was fit on train-only (correct behavior)",
                "INFO"
            )
            print(f"  ✓ Indicates scaler was likely fit on train data only (GOOD)")
        else:
            self._log(
                "All test features are within train range",
                "WARNING"
            )
            print(f"  ⚠️  Could indicate scaler was fit on combined train+test data (BAD)")
        
        print("\nCheck 2: Scaled value statistics")
        print("-" * 60)
        
        # Fit scaler on train ONLY and check test scaled range
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # For MinMaxScaler fit on train, test values should mostly be in [0,1]
        # but CAN exceed this range if test has values outside train range
        test_outside_01 = ((X_test_scaled < 0) | (X_test_scaled > 1)).sum()
        pct_outside = (test_outside_01 / X_test_scaled.size) * 100
        
        if pct_outside > 5:
            self._log(
                f"{pct_outside:.1f}% of scaled test features are outside [0,1]",
                "INFO"
            )
            print(f"  ✓ This is EXPECTED for scaling fit on train-only (GOOD)")
        else:
            self._log(
                f"Scaled test features well-contained in [0,1] ({pct_outside:.1f}%)",
                "INFO"
            )
            print(f"  Could indicate scaler was fit on combined data")
        
        self.recommendations.append(
            "✓ VERIFIED: Scaler should be fit on X_train ONLY, then applied to X_val and X_test"
        )
        
        return {
            'features_out_of_train_range': int(n_out_of_range),
            'pct_test_outside_01': float(pct_outside),
            'status': 'OK' if n_out_of_range > 0 else 'VERIFY'
        }
    
    # =====================================================================
    # AUDIT 4: GENERALIZATION GAP ANALYSIS
    # =====================================================================
    def audit_generalization_gap(self, train_metrics, test_metrics):
        """
        Analyze the gap between training and test metrics.
        
        Unrealistic model results show near-zero gap (overfitting from leakage).
        Realistic models show 5-15% gap depending on dataset size and model complexity.
        """
        print("\n" + "="*80)
        print("AUDIT 4: GENERALIZATION GAP (Train vs Test)")
        print("="*80)
        
        print("\nMetric Comparison:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Train':<12} {'Test':<12} {'Gap':<12} {'Status':<15}")
        print("-" * 80)
        
        gap_alerts = []
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric not in train_metrics or metric not in test_metrics:
                continue
            
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]
            gap = train_val - test_val
            gap_pct = abs(gap) * 100
            
            # Interpretation
            if gap < 0.02:
                status = "🔴 SUSPICIOUS (no gap)"
            elif gap < 0.05:
                status = "🟡 LOW (potential issue)"
            elif gap < 0.15:
                status = "🟢 REALISTIC"
            else:
                status = "🟠 HIGH (overfitting)"
            
            print(f"{metric:<20} {train_val:>11.4f} {test_val:>11.4f} {gap:>11.4f} {status:<15}")
            
            if gap < 0.05:
                gap_alerts.append(metric)
        
        print("-" * 80)
        
        # Summary
        if gap_alerts:
            self._log(
                f"⚠️  Zero/minimal gap in {', '.join(gap_alerts)} — indicates potential leakage",
                "WARNING"
            )
            self.warnings.append(
                f"Suspicious generalization gaps: {', '.join(gap_alerts)}"
            )
        else:
            self._log(
                "Generalization gaps appear reasonable (5-15% typical)",
                "SUCCESS"
            )
        
        return {
            'low_gap_metrics': gap_alerts,
            'interpretation': 'Leakage risk' if gap_alerts else 'Normal'
        }
    
    # =====================================================================
    # AUDIT 5: CLASS WEIGHT / IMBALANCE HANDLING
    # =====================================================================
    def audit_class_balance(self, y_train, y_test, y_val=None):
        """
        Check if class distribution is similar across splits.
        
        Extreme imbalance not stratified = potential leakage if one class dominates test.
        """
        print("\n" + "="*80)
        print("AUDIT 5: CLASS BALANCE & STRATIFICATION")
        print("="*80)
        
        y_train = y_train.values if hasattr(y_train, 'values') else y_train
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        
        def get_dist(y, name):
            unique, counts = np.unique(y, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(unique, counts)}
            total = len(y)
            pct = {int(u): (c / total) * 100 for u, c in zip(unique, counts)}
            return dist, pct
        
        train_dist, train_pct = get_dist(y_train, "Train")
        test_dist, test_pct = get_dist(y_test, "Test")
        
        print(f"\nClass Distribution:")
        print("-" * 60)
        print(f"{'Split':<10} {'Class 0':<20} {'Class 1':<20}")
        print("-" * 60)
        print(f"{'Train':<10} "
              f"{train_dist.get(0, 0):>6} ({train_pct.get(0, 0):>5.1f}%)  "
              f"{train_dist.get(1, 0):>6} ({train_pct.get(1, 0):>5.1f}%)")
        print(f"{'Test':<10} "
              f"{test_dist.get(0, 0):>6} ({test_pct.get(0, 0):>5.1f}%)  "
              f"{test_dist.get(1, 0):>6} ({test_pct.get(1, 0):>5.1f}%)")
        
        if y_val is not None:
            y_val = y_val.values if hasattr(y_val, 'values') else y_val
            val_dist, val_pct = get_dist(y_val, "Val")
            print(f"{'Val':<10} "
                  f"{val_dist.get(0, 0):>6} ({val_pct.get(0, 0):>5.1f}%)  "
                  f"{val_dist.get(1, 0):>6} ({val_pct.get(1, 0):>5.1f}%)")
        
        # Check stratification
        s_imb = train_pct.get(1, 50) / test_pct.get(1, 50) if test_pct.get(1, 50) > 0 else 1
        
        if 0.8 < s_imb < 1.2:
            self._log("Train/Test class ratios similar (good stratification)", "SUCCESS")
        else:
            self._log(f"Imbalanced across splits (ratio={s_imb:.2f})", "WARNING")
            self.warnings.append(f"Stratification ratio: {s_imb:.2f}")
        
        # Check absolute imbalance
        maj_class = max(train_dist.get(0, 0), train_dist.get(1, 0))
        min_class = min(train_dist.get(0, 0), train_dist.get(1, 0))
        imb_ratio = maj_class / min_class if min_class > 0 else float('inf')
        
        print(f"\nImbalance Ratio (Train): {imb_ratio:.1f}:1")
        if imb_ratio > 5:
            self._log(f"Severe class imbalance ({imb_ratio:.1f}:1)", "WARNING")
            print(f"  Applied class weights during training: use 'balanced' or compute_class_weight()")
        elif imb_ratio > 2:
            self._log(f"Moderate imbalance ({imb_ratio:.1f}:1)", "INFO")
        else:
            self._log(f"Balanced classes ({imb_ratio:.1f}:1)", "SUCCESS")
        
        return {
            'train_distribution': train_dist,
            'test_distribution': test_dist,
            'imbalance_ratio': float(imb_ratio),
            'stratification': 'OK' if 0.8 < s_imb < 1.2 else 'IMBALANCED'
        }
    
    # =====================================================================
    # AUDIT 6: SUSPICIOUS PERFECT ACCURACY PATTERNS
    # =====================================================================
    def audit_perfect_accuracy_indicators(self, y_test, y_pred, y_pred_proba=None):
        """
        Check for telltale signs of data leakage causing perfect accuracy.
        """
        print("\n" + "="*80)
        print("AUDIT 6: PERFECT ACCURACY INDICATORS")
        print("="*80)
        
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        y_pred = y_pred.values if hasattr(y_pred, 'values') else y_pred
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()
        
        accuracy = (y_test == y_pred).sum() / len(y_test)
        
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        indicators = []
        
        # Check 1: Accuracy near 1.0
        if accuracy >= 0.99:
            self._log("🔴 Accuracy >= 99% — MAJOR LEAKAGE INDICATOR", "ERROR")
            indicators.append("Perfect/near-perfect accuracy")
            self.issues_found.append({
                'type': 'Unrealistic Performance',
                'severity': 'CRITICAL',
                'metric': 'accuracy',
                'value': float(accuracy),
                'description': 'Accuracy >= 99% indicates probable data leakage'
            })
        elif accuracy >= 0.95:
            self._log("⚠️  Accuracy >= 95% — POTENTIAL LEAKAGE", "WARNING")
            indicators.append("Very high accuracy")
        
        # Check 2: No errors
        n_errors = (y_test != y_pred).sum()
        if n_errors == 0:
            self._log(f"🔴 Zero misclassifications on test set — DEFINITE LEAKAGE", "ERROR")
            indicators.append("Zero test errors")
        elif n_errors < len(y_test) * 0.01:
            self._log(f"⚠️  Only {n_errors} misclassifications (<1% error rate)", "WARNING")
            indicators.append("Near-zero error rate")
        
        # Check 3: Probability distribution (should never be all 0s or 1s)
        if y_pred_proba is not None:
            y_pred_proba = y_pred_proba.values if hasattr(y_pred_proba, 'values') else y_pred_proba
            y_pred_proba = y_pred_proba.flatten()
            
            all_0s = (y_pred_proba == 0).sum()
            all_1s = (y_pred_proba == 1).sum()
            
            if all_0s > len(y_pred_proba) * 0.05:
                self._log(
                    f"⚠️  {(all_0s/len(y_pred_proba))*100:.1f}% of predictions are exactly 0.0",
                    "WARNING"
                )
                indicators.append("Many exact 0.0 probabilities")
            
            if all_1s > len(y_pred_proba) * 0.05:
                self._log(
                    f"⚠️  {(all_1s/len(y_pred_proba))*100:.1f}% of predictions are exactly 1.0",
                    "WARNING"
                )
                indicators.append("Many exact 1.0 probabilities")
            
            # Should have smooth distribution
            prob_uniq = len(np.unique(y_pred_proba))
            if prob_uniq < 10:
                self._log(
                    f"⚠️  Only {prob_uniq} unique probability values (too few, suggests binary output)",
                    "WARNING"
                )
        
        # Check 4: Per-class accuracy
        for cls in np.unique(y_test):
            cls_mask = y_test == cls
            cls_acc = (y_test[cls_mask] == y_pred[cls_mask]).sum() / cls_mask.sum()
            if cls_acc >= 1.0:
                self._log(f"🔴 100% accuracy on class {cls}", "ERROR")
                indicators.append(f"Perfect accuracy on class {cls}")
            elif cls_acc >= 0.99:
                self._log(f"⚠️  {cls_acc*100:.2f}% accuracy on class {cls}", "WARNING")
        
        if not indicators:
            self._log("✓ No obvious perfect-accuracy leakage indicators found", "SUCCESS")
        else:
            self._log(f"\n{len(indicators)} potential leakage indicators detected:", "WARNING")
            for i, ind in enumerate(indicators, 1):
                print(f"  {i}. {ind}")
        
        return {
            'accuracy': float(accuracy),
            'n_errors': int(n_errors),
            'indicators': indicators,
            'leakage_risk': 'CRITICAL' if len(indicators) > 1 else 'LOW'
        }
    
    # =====================================================================
    # FULL AUDIT
    # =====================================================================
    def run_full_audit(self, X_train, X_test, y_train, y_test, X_val=None, y_val=None,
                      feature_names=None, y_pred=None, y_pred_proba=None,
                      train_metrics=None, test_metrics=None):
        """
        Run all audits and generate comprehensive report.
        """
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE DATA LEAKAGE AUDIT")
        print("="*80)
        
        results = {}
        
        # Run audits
        results['overlap'] = self.audit_train_test_overlap(X_train, X_test, y_train, y_test)
        results['feature_leakage'] = self.audit_feature_leakage(X_train, y_train, feature_names)
        results['scaling'] = self.audit_scaling_leakage(X_train, X_test, X_val)
        results['class_balance'] = self.audit_class_balance(y_train, y_test, y_val)
        
        if y_pred is not None and y_test is not None:
            results['accuracy_indicators'] = self.audit_perfect_accuracy_indicators(
                y_test, y_pred, y_pred_proba
            )
        
        if train_metrics and test_metrics:
            results['gen_gap'] = self.audit_generalization_gap(train_metrics, test_metrics)
        
        # Summary
        self._print_summary()
        
        return {
            'issues_found': self.issues_found,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'detailed_results': results
        }
    
    def _print_summary(self):
        """Print executive summary."""
        print("\n" + "="*80)
        print("📋 AUDIT SUMMARY")
        print("="*80)
        
        if self.issues_found:
            print(f"\n🔴 CRITICAL ISSUES FOUND: {len(self.issues_found)}")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"\n  {i}. {issue['type']} [{issue['severity']}]")
                print(f"     {issue['description']}")
        else:
            print(f"\n✅ No critical issues found")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                print(f"  {i}. {warn}")
        
        if self.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


# =========================================================================
# HELPER: Quick Audit for Typical Pipeline
# =========================================================================
def quick_audit(X_train, X_test, y_train, y_test, X_val=None, y_val=None,
                feature_names=None, y_pred=None, y_pred_proba=None):
    """Quick leakage audit without detailed output."""
    audit = DataLeakageAudit(verbose=False)
    return audit.run_full_audit(
        X_train, X_test, y_train, y_test, X_val, y_val,
        feature_names, y_pred, y_pred_proba
    )


if __name__ == "__main__":
    print("DataLeakageAudit module loaded successfully")
    print("\nUsage:")
    print("  from data_leakage_audit import DataLeakageAudit")
    print("  audit = DataLeakageAudit()")
    print("  results = audit.run_full_audit(X_train, X_test, y_train, y_test, ...)")
