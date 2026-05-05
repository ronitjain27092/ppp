"""
COMPREHENSIVE MODEL SANITY CHECK & DEBUGGING SCRIPT
====================================================

Diagnoses why ML models show near-perfect accuracy (~100%) and provides fixes.

Tests performed:
1. Random Label Test - Model trained on shuffled labels (should get ~50% accuracy)
2. Feature Shuffling Test - Shuffle features and retest (should drop significantly)
3. Correlation Analysis - Check which features are leaky
4. Duplicate Detection - Find duplicate/near-duplicate samples
5. Train-Test Contamination - Verify proper data separation
6. Feature Ablation - Remove top features one by one
7. Noise Injection - Add noise and test robustness
8. Proper Train-Test Split - Test with correct methodology
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import hashlib
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModelSanityCheck:
    """Comprehensive model debugging and sanity checks."""
    
    def __init__(self, model, X, y, feature_names=None, test_size=0.2, random_state=42):
        """
        Initialize sanity check.
        
        Args:
            model: Trained sklearn model with predict/predict_proba methods
            X: Feature matrix
            y: Target labels
            feature_names: Feature names (optional)
            test_size: Test set size
            random_state: Random seed
        """
        self.model = model
        self.X = X.values if hasattr(X, 'values') else X
        self.y = y.values if hasattr(y, 'values') else y
        self.y = self.y.flatten()
        self.feature_names = feature_names or [f'feature_{i}' for i in range(self.X.shape[1])]
        self.test_size = test_size
        self.random_state = random_state
        
        self.results = {}
        self.warnings = []
        self.recommendations = []
        
        print("\n" + "="*80)
        print("🔍 MODEL SANITY CHECK & LEAKAGE DETECTION")
        print("="*80)
        
    # =========================================================================
    # TEST 1: RANDOM LABEL TEST (Critical sanity check)
    # =========================================================================
    def test_random_labels(self):
        """
        Train model on completely shuffled labels.
        
        Expected: Accuracy should be ~50% (random guessing)
        Problem: If accuracy stays high, model is memorizing or overfitting severely
        """
        print("\n" + "="*80)
        print("TEST 1: RANDOM LABEL TEST")
        print("="*80)
        print("Training model on shuffled labels (should get ~50% accuracy)...\n")
        
        # Create random labels
        y_random = np.random.randint(0, 2, len(self.y))
        
        # Split with CORRECT methodology
        X_train, X_test, y_train_random, y_test_random = train_test_split(
            self.X, y_random, test_size=self.test_size, random_state=self.random_state,
            stratify=y_random
        )
        
        # Train model on random labels
        model_random = self._train_model_copy(X_train, y_train_random)
        
        # Test
        y_pred_random = model_random.predict(X_test)
        acc_random = accuracy_score(y_test_random, y_pred_random)
        
        print(f"Accuracy on random labels: {acc_random:.4f}")
        
        # Analysis
        if acc_random < 0.55:
            print(f"✅ GOOD: Model achieves {acc_random:.1%} on random labels (random guessing)")
            status = "PASS"
        elif acc_random < 0.70:
            print(f"⚠️  WARNING: Model achieves {acc_random:.1%} on random labels (some memorization)")
            self.warnings.append(f"Model shows {acc_random:.1%} accuracy on shuffled labels (expected ~50%)")
            status = "WARNING"
        else:
            print(f"🔴 CRITICAL: Model achieves {acc_random:.1%} on random labels!")
            print(f"   This indicates severe overfitting/leakage - model is memorizing data")
            self.warnings.append(f"CRITICAL: {acc_random:.1%} accuracy on random labels = severe leakage")
            status = "CRITICAL"
        
        self.results['random_label_test'] = {
            'accuracy': float(acc_random),
            'status': status
        }
        
        return acc_random
    
    # =========================================================================
    # TEST 2: FEATURE SHUFFLING TEST
    # =========================================================================
    def test_feature_shuffling(self):
        """
        Shuffle features and test accuracy drop.
        
        Expected: Significant accuracy drop (>20%)
        Problem: Small drop = features don't matter (leakage)
        """
        print("\n" + "="*80)
        print("TEST 2: FEATURE SHUFFLING TEST")
        print("="*80)
        print("Shuffling features and retesting (should see big accuracy drop)...\n")
        
        # Get baseline accuracy on test set
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y
        )
        
        y_pred_baseline = self.model.predict(X_test)
        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        
        print(f"Baseline accuracy: {acc_baseline:.4f}")
        
        # Shuffle all features
        X_test_shuffled = X_test.copy()
        for col in range(X_test_shuffled.shape[1]):
            np.random.shuffle(X_test_shuffled[:, col])
        
        y_pred_shuffled = self.model.predict(X_test_shuffled)
        acc_shuffled = accuracy_score(y_test, y_pred_shuffled)
        
        drop = acc_baseline - acc_shuffled
        drop_pct = drop * 100
        
        print(f"Accuracy after shuffling: {acc_shuffled:.4f}")
        print(f"Drop: {drop_pct:.1f}%")
        
        # Analysis
        if drop_pct > 20:
            print(f"✅ GOOD: {drop_pct:.1f}% accuracy drop from shuffling (features matter)")
            status = "PASS"
        elif drop_pct > 10:
            print(f"⚠️  WARNING: Only {drop_pct:.1f}% drop (features may not be critical)")
            status = "WARNING"
        else:
            print(f"🔴 CRITICAL: Only {drop_pct:.1f}% drop from shuffling!")
            print(f"   Model works without features = pure memorization/leakage")
            status = "CRITICAL"
        
        self.results['feature_shuffling_test'] = {
            'baseline_accuracy': float(acc_baseline),
            'shuffled_accuracy': float(acc_shuffled),
            'drop_percent': float(drop_pct),
            'status': status
        }
        
        return drop_pct
    
    # =========================================================================
    # TEST 3: CORRELATION ANALYSIS (Feature leakage detection)
    # =========================================================================
    def test_feature_correlation(self, threshold=0.95):
        """
        Identify features suspiciously correlated with target.
        
        Expected: Max correlation < 0.80
        Problem: High correlation (>0.95) = feature encodes the target
        """
        print("\n" + "="*80)
        print("TEST 3: FEATURE CORRELATION ANALYSIS")
        print("="*80)
        print(f"Checking feature-target correlations (threshold: {threshold})...\n")
        
        correlations = {}
        for i, feat_name in enumerate(self.feature_names):
            corr = np.corrcoef(self.X[:, i], self.y)[0, 1]
            correlations[feat_name] = abs(corr) if not np.isnan(corr) else 0
        
        sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top 10 features by correlation with target:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Feature':<30} {'|Correlation|':<15} {'Status':<10}")
        print("-" * 70)
        
        leaky_features = []
        for rank, (feat, corr) in enumerate(sorted_corrs[:10], 1):
            status = "🔴 LEAKY" if corr > threshold else "🟡 NOTABLE" if corr > 0.70 else "✅ OK"
            print(f"{rank:<5} {feat:<30} {corr:>14.4f}  {status}")
            
            if corr > threshold:
                leaky_features.append((feat, corr))
        
        print("-" * 70)
        
        if leaky_features:
            print(f"\n🔴 {len(leaky_features)} LEAKY FEATURES DETECTED (>0.95 correlation):")
            for feat, corr in leaky_features:
                print(f"   - {feat}: {corr:.4f} ← DROP THIS FEATURE")
            self.recommendations.append(f"🔴 DROP these features: {[f for f, _ in leaky_features]}")
            status = "CRITICAL"
        else:
            print(f"\n✅ No leaky features detected (all correlations < {threshold})")
            status = "PASS"
        
        self.results['feature_correlation'] = {
            'leaky_features': [(f, float(c)) for f, c in leaky_features],
            'top_correlations': {f: float(c) for f, c in sorted_corrs[:5]},
            'status': status
        }
        
        return leaky_features
    
    # =========================================================================
    # TEST 4: DUPLICATE DETECTION
    # =========================================================================
    def test_duplicate_samples(self):
        """
        Detect duplicate or near-duplicate samples.
        
        Expected: 0-1 duplicates (rare by chance)
        Problem: Many duplicates = data leakage across train/test
        """
        print("\n" + "="*80)
        print("TEST 4: DUPLICATE SAMPLE DETECTION")
        print("="*80)
        print("Checking for duplicate/near-duplicate samples...\n")
        
        # Exact duplicates
        X_df = pd.DataFrame(self.X)
        duplicates = X_df.duplicated().sum()
        
        print(f"Exact duplicates in dataset: {duplicates}")
        if duplicates > 0:
            print(f"  🔴 WARNING: {duplicates} exact duplicate rows found!")
            self.warnings.append(f"{duplicates} exact duplicate rows")
        else:
            print(f"  ✅ No exact duplicates")
        
        # Near-duplicates (rounded to 4 decimals)
        def hash_row(row, decimals=4):
            rounded = np.round(row, decimals)
            return hashlib.md5(rounded.tobytes()).hexdigest()
        
        hashes = {}
        near_dups = 0
        for i, row in enumerate(self.X):
            h = hash_row(row)
            if h in hashes:
                near_dups += 1
            else:
                hashes[h] = i
        
        print(f"Near-duplicates (4-decimal precision): {near_dups}")
        if near_dups > len(self.X) * 0.05:  # >5% near-duplicate
            print(f"  🔴 WARNING: High near-duplicate rate ({near_dups/len(self.X)*100:.1f}%)")
            self.warnings.append(f"{near_dups} near-duplicate samples ({near_dups/len(self.X)*100:.1f}%)")
            status = "WARNING"
        else:
            print(f"  ✅ Near-duplicate rate acceptable")
            status = "PASS"
        
        # Train/Test split check
        print(f"\nTrain/Test contamination check:")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y
        )
        
        train_hashes = {hash_row(row, decimals=4) for row in X_train}
        test_contamination = sum(1 for row in X_test if hash_row(row, decimals=4) in train_hashes)
        
        print(f"  Test samples also in training: {test_contamination}")
        if test_contamination > 0:
            print(f"  🔴 CRITICAL: {test_contamination} test samples are in training set!")
            self.warnings.append(f"CRITICAL: {test_contamination} test-train contamination")
            status = "CRITICAL"
        else:
            print(f"  ✅ No train/test overlap")
        
        self.results['duplicate_detection'] = {
            'exact_duplicates': int(duplicates),
            'near_duplicates': int(near_dups),
            'train_test_contamination': int(test_contamination),
            'status': status
        }
        
        return {'exact': duplicates, 'near': near_dups, 'contamination': test_contamination}
    
    # =========================================================================
    # TEST 5: TRAINING DATA LEAKAGE
    # =========================================================================
    def test_train_test_leakage(self):
        """
        Test if model was trained on test data.
        
        Proper:
          - Split BEFORE preprocessing/scaling
          - Scaler fit ONLY on training data
          - No test data statistics in training
        """
        print("\n" + "="*80)
        print("TEST 5: TRAIN-TEST SPLIT METHODOLOGY CHECK")
        print("="*80)
        print("Verifying proper train-test separation...\n")
        
        # Method 1: Train on proper split, evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y
        )
        
        # Use StandardScaler (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train fresh model
        model_proper = self._train_model_copy(X_train_scaled, y_train)
        
        # Test
        y_pred_proper = model_proper.predict(X_test_scaled)
        acc_proper = accuracy_score(y_test, y_pred_proper)
        
        print(f"Accuracy with PROPER train-test split:")
        print(f"  Scaler fit on training data only: {acc_proper:.4f}")
        
        # Compare with original model
        try:
            y_pred_original = self.model.predict(self.X[:len(X_test)])
            acc_original = accuracy_score(self.y[:len(X_test)], y_pred_original)
            print(f"  Original model accuracy: {acc_original:.4f}")
            
            if acc_original > 0.99 and acc_proper < 0.90:
                print(f"\n🔴 CRITICAL LEAKAGE DETECTED!")
                print(f"   Original: {acc_original:.1%} vs Proper split: {acc_proper:.1%}")
                print(f"   Model was trained on test data or scaler fit on combined data")
                status = "CRITICAL"
            else:
                print(f"\n✅ Accuracies similar - methodology appears OK")
                status = "PASS"
        except:
            status = "UNKNOWN"
        
        self.results['train_test_leakage'] = {
            'proper_methodology_accuracy': float(acc_proper),
            'status': status
        }
        
        return acc_proper
    
    # =========================================================================
    # TEST 6: FEATURE ABLATION
    # =========================================================================
    def test_feature_ablation(self, n_features=10):
        """
        Remove top features one by one and measure accuracy drop.
        
        Expected: Removing top features should drop accuracy
        Problem: No drop = features don't matter
        """
        print("\n" + "="*80)
        print("TEST 6: FEATURE ABLATION")
        print("="*80)
        print(f"Removing top {n_features} features one by one...\n")
        
        # Get baseline
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y
        )
        
        y_pred_baseline = self.model.predict(X_test)
        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        
        print(f"Baseline accuracy: {acc_baseline:.4f}\n")
        print(f"{'Feature':<30} {'Accuracy w/o Feature':<20} {'Drop':<10}")
        print("-" * 60)
        
        ablation_results = {}
        for i in range(min(n_features, len(self.feature_names))):
            # Remove feature i
            X_test_ablated = np.delete(X_test, i, axis=1)
            try:
                y_pred_ablated = self.model.predict(X_test_ablated)
                acc_ablated = accuracy_score(y_test, y_pred_ablated)
            except:
                # If deletion breaks model, skip
                acc_ablated = acc_baseline
            
            drop = acc_baseline - acc_ablated
            ablation_results[self.feature_names[i]] = drop
            
            print(f"{self.feature_names[i]:<30} {acc_ablated:>18.4f} {drop*100:>9.1f}%")
        
        print("-" * 60)
        
        max_drop = max(ablation_results.values())
        
        if max_drop > 0.10:
            print(f"\n✅ Feature ablation shows impact (max drop: {max_drop*100:.1f}%)")
            status = "PASS"
        else:
            print(f"\n🔴 Features have minimal impact (max drop: {max_drop*100:.1f}%)")
            print(f"   This suggests features don't matter or leakage exists")
            status = "CRITICAL"
        
        self.results['feature_ablation'] = {
            'ablation_results': {k: float(v) for k, v in ablation_results.items()},
            'max_drop': float(max_drop),
            'status': status
        }
        
        return ablation_results
    
    # =========================================================================
    # TEST 7: NOISE INJECTION
    # =========================================================================
    def test_noise_injection(self):
        """
        Add Gaussian noise to features and retest.
        
        Expected: Accuracy drops gradually with increasing noise
        Problem: Stays high = overfitting
        """
        print("\n" + "="*80)
        print("TEST 7: NOISE INJECTION TEST")
        print("="*80)
        print("Adding noise to features and measuring accuracy drop...\n")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y
        )
        
        y_pred_baseline = self.model.predict(X_test)
        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        
        noise_levels = np.linspace(0, 1.0, 11)
        results = {'noise': [], 'accuracy': []}
        
        print(f"{'Noise Level':<15} {'Accuracy':<15} {'Drop':<10}")
        print("-" * 40)
        print(f"{'0.0':<15} {acc_baseline:>14.4f} {0:>9.1f}%")
        
        for noise_level in noise_levels[1:]:
            X_test_noisy = X_test + np.random.normal(0, noise_level * np.std(X_test), X_test.shape)
            
            y_pred_noisy = self.model.predict(X_test_noisy)
            acc_noisy = accuracy_score(y_test, y_pred_noisy)
            drop = (acc_baseline - acc_noisy) * 100
            
            results['noise'].append(noise_level)
            results['accuracy'].append(acc_noisy)
            
            print(f"{noise_level:<15.1f} {acc_noisy:>14.4f} {drop:>9.1f}%")
        
        # Check robustness at 50% noise
        acc_at_50 = results['accuracy'][5] if len(results['accuracy']) > 5 else results['accuracy'][-1]
        drop_at_50 = (acc_baseline - acc_at_50) * 100
        
        print("-" * 40)
        
        if drop_at_50 > 15:
            print(f"\n✅ Model degrades gracefully with noise (drop at 50% noise: {drop_at_50:.1f}%)")
            status = "PASS"
        else:
            print(f"\n🔴 Model is robust to noise (drop at 50% noise: {drop_at_50:.1f}%)")
            print(f"   This suggests overfitting or leakage")
            status = "CRITICAL"
        
        self.results['noise_injection'] = {
            'noise_levels': [float(n) for n in results['noise']],
            'accuracies': [float(a) for a in results['accuracy']],
            'drop_at_50_percent': float(drop_at_50),
            'status': status
        }
        
        return results
    
    # =========================================================================
    # TEST 8: PROPER TRAIN-VAL-TEST SPLIT
    # =========================================================================
    def test_proper_three_way_split(self):
        """
        Test with proper 70/15/15 train/val/test methodology.
        """
        print("\n" + "="*80)
        print("TEST 8: PROPER 3-WAY SPLIT (70/15/15)")
        print("="*80)
        print("Training with strictly separated train/val/test sets...\n")
        
        # Split: 70/30 first
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=0.15, random_state=self.random_state, stratify=self.y
        )
        
        # Split temp into train/val: 70/(70+15) ≈ 0.82 of temp
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(self.X)*100:.0f}%)")
        print(f"Val set:   {X_val.shape[0]} samples ({X_val.shape[0]/len(self.X)*100:.0f}%)")
        print(f"Test set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(self.X)*100:.0f}%)")
        
        # Train model on proper split
        model_split = self._train_model_copy(X_train, y_train)
        
        # Evaluate on all three sets
        y_pred_train = model_split.predict(X_train)
        y_pred_val = model_split.predict(X_val)
        y_pred_test = model_split.predict(X_test)
        
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_val = accuracy_score(y_val, y_pred_val)
        acc_test = accuracy_score(y_test, y_pred_test)
        
        print(f"\nResults:")
        print(f"  Train Accuracy: {acc_train:.4f}")
        print(f"  Val Accuracy:   {acc_val:.4f}")
        print(f"  Test Accuracy:  {acc_test:.4f}")
        
        # Analysis
        gap_val_test = abs(acc_val - acc_test)
        gap_train_test = acc_train - acc_test
        
        print(f"\nGeneralization Gap (Train - Test): {gap_train_test*100:.1f}%")
        
        if gap_train_test < 0.02:
            print(f"  🔴 CRITICAL: Almost no gap! Possible leakage")
            status = "CRITICAL"
        elif gap_train_test < 0.05:
            print(f"  🟡 WARNING: Very small gap (<5%)")
            status = "WARNING"
        elif gap_train_test > 0.20:
            print(f"  🟡 WARNING: Large gap (>20%) - model may be underfitting")
            status = "WARNING"
        else:
            print(f"  ✅ GOOD: Realistic gap (5-20%)")
            status = "PASS"
        
        self.results['three_way_split'] = {
            'train_accuracy': float(acc_train),
            'val_accuracy': float(acc_val),
            'test_accuracy': float(acc_test),
            'generalization_gap': float(gap_train_test),
            'status': status
        }
        
        return {'train': acc_train, 'val': acc_val, 'test': acc_test}
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    def _train_model_copy(self, X, y):
        """Train a fresh copy of the model."""
        if isinstance(self.model, LogisticRegression):
            m = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif isinstance(self.model, RandomForestClassifier):
            m = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            # Generic sklearn model
            m = self.model.__class__(random_state=self.random_state)
        
        m.fit(X, y)
        return m
    
    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    def generate_report(self):
        """Generate comprehensive report."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE SANITY CHECK REPORT")
        print("="*80)
        
        print("\n1️⃣  RANDOM LABEL TEST:")
        rl_result = self.results.get('random_label_test', {})
        print(f"   Accuracy on shuffled labels: {rl_result.get('accuracy', 0):.4f}")
        print(f"   Status: {rl_result.get('status', 'UNKNOWN')}")
        
        print("\n2️⃣  FEATURE SHUFFLING TEST:")
        fs_result = self.results.get('feature_shuffling_test', {})
        print(f"   Baseline accuracy: {fs_result.get('baseline_accuracy', 0):.4f}")
        print(f"   Shuffled accuracy: {fs_result.get('shuffled_accuracy', 0):.4f}")
        print(f"   Drop: {fs_result.get('drop_percent', 0):.1f}%")
        print(f"   Status: {fs_result.get('status', 'UNKNOWN')}")
        
        print("\n3️⃣  FEATURE CORRELATION:")
        fc_result = self.results.get('feature_correlation', {})
        leaky = fc_result.get('leaky_features', [])
        print(f"   Leaky features (>0.95): {len(leaky)}")
        if leaky:
            for feat, corr in leaky[:3]:
                print(f"     - {feat}: {corr:.4f}")
        print(f"   Status: {fc_result.get('status', 'UNKNOWN')}")
        
        print("\n4️⃣  DUPLICATE DETECTION:")
        dup_result = self.results.get('duplicate_detection', {})
        print(f"   Exact duplicates: {dup_result.get('exact_duplicates', 0)}")
        print(f"   Near-duplicates: {dup_result.get('near_duplicates', 0)}")
        print(f"   Train-test contamination: {dup_result.get('train_test_contamination', 0)}")
        print(f"   Status: {dup_result.get('status', 'UNKNOWN')}")
        
        print("\n5️⃣  TRAIN-TEST SPLIT METHOD:")
        ttl_result = self.results.get('train_test_leakage', {})
        print(f"   Accuracy (proper methodology): {ttl_result.get('proper_methodology_accuracy', 0):.4f}")
        print(f"   Status: {ttl_result.get('status', 'UNKNOWN')}")
        
        print("\n6️⃣  FEATURE ABLATION:")
        abl_result = self.results.get('feature_ablation', {})
        print(f"   Max accuracy drop: {abl_result.get('max_drop', 0)*100:.1f}%")
        print(f"   Status: {abl_result.get('status', 'UNKNOWN')}")
        
        print("\n7️⃣  NOISE INJECTION:")
        noise_result = self.results.get('noise_injection', {})
        print(f"   Drop at 50% noise: {noise_result.get('drop_at_50_percent', 0):.1f}%")
        print(f"   Status: {noise_result.get('status', 'UNKNOWN')}")
        
        print("\n8️⃣  PROPER 3-WAY SPLIT:")
        split_result = self.results.get('three_way_split', {})
        print(f"   Train: {split_result.get('train_accuracy', 0):.4f}")
        print(f"   Val:   {split_result.get('val_accuracy', 0):.4f}")
        print(f"   Test:  {split_result.get('test_accuracy', 0):.4f}")
        print(f"   Gap:   {split_result.get('generalization_gap', 0)*100:.1f}%")
        print(f"   Status: {split_result.get('status', 'UNKNOWN')}")
        
        # Warnings
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Recommendations
        if self.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Overall diagnosis
        critical_count = sum(1 for r in self.results.values() if r.get('status') == 'CRITICAL')
        
        print("\n" + "="*80)
        if critical_count == 0:
            print("✅ DIAGNOSIS: Model appears healthy (no critical issues)")
        elif critical_count == 1:
            print("⚠️  DIAGNOSIS: Some issues found - review recommendations")
        else:
            print(f"🔴 DIAGNOSIS: SEVERE ISSUES ({critical_count} critical) - fix immediately")
        print("="*80 + "\n")


def run_sanity_check(model, X, y, feature_names=None):
    """
    Run complete sanity check on model.
    
    Args:
        model: Trained sklearn model
        X: Feature matrix
        y: Target labels
        feature_names: Optional feature names
    """
    checker = ModelSanityCheck(model, X, y, feature_names)
    
    # Run all tests
    checker.test_random_labels()
    checker.test_feature_shuffling()
    leaky_features = checker.test_feature_correlation()
    checker.test_duplicate_samples()
    checker.test_train_test_leakage()
    checker.test_feature_ablation()
    checker.test_noise_injection()
    checker.test_proper_three_way_split()
    
    # Generate report
    checker.generate_report()
    
    return checker


if __name__ == "__main__":
    print("Model Sanity Check module loaded successfully")
    print("\nUsage:")
    print("  from model_sanity_check import run_sanity_check")
    print("  checker = run_sanity_check(model, X, y, feature_names)")
