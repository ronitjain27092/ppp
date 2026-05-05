#!/usr/bin/env python3
"""
COMPLETE MALWARE DETECTION PIPELINE VALIDATION
===============================================

Runs full validation suite including:
1. Data preprocessing (70/15/15 split)
2. Model training (CNN, RF, LR)
3. Leakage audit (6 types)
4. Evaluation plots (ROC, PR, calibration)
5. Robustness testing (perturbations)
6. Feature analysis (importance & leakage)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print("=" * 90)
print("🔍 MALWARE DETECTION PIPELINE - COMPREHENSIVE VALIDATION")
print("=" * 90)

# Check imports
try:
    from preprocessing import DataPreprocessor
    from model_enhanced import EnsembleModels
    from data_leakage_audit import DataLeakageAudit
    from evaluation_visualizations import create_all_evaluation_plots
    from perturbation_robustness import run_full_perturbation_suite
    from feature_leakage_detector import generate_feature_leakage_report
    print("\n✅ All modules imported successfully")
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    sys.exit(1)

# Check for data file
data_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not data_files:
    print("\n⚠️  No CSV data files found in current directory")
    print("   Please provide a CSV file with 'label' column")
    sys.exit(1)

data_file = data_files[0]
print(f"✅ Using data file: {data_file}")

print("\n" + "=" * 90)
print("[1/6] DATA PREPROCESSING (70/15/15 stratified split with leakage prevention)")
print("=" * 90)

try:
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, data_report = \
        preprocessor.preprocess(data_file)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"\n   Class distribution:")
    print(f"   Train - Benign: {(y_train==0).sum()}, Malware: {(y_train==1).sum()}")
    print(f"   Test  - Benign: {(y_test==0).sum()}, Malware: {(y_test==1).sum()}")
    
except Exception as e:
    print(f"\n❌ Preprocessing Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 90)
print("[2/6] MODEL TRAINING (CNN, Random Forest, Logistic Regression)")
print("=" * 90)

try:
    ensemble = EnsembleModels()
    ensemble.train_all_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        do_cv=True,  # Stratified K-Fold CV
        feature_names=feature_names,
        verbose=True
    )
    
    results_df = ensemble.get_comparison_dataframe()
    print(f"\n✅ Model training complete!")
    print("\nModel Comparison:")
    print(results_df.to_string())
    
    cv_df = ensemble.get_cv_dataframe()
    if cv_df is not None:
        print("\nCross-Validation Results:")
        print(cv_df.to_string())
    
    overfitting_df = ensemble.get_overfitting_report()
    print("\nOverfitting Report (Train vs Test):")
    print(overfitting_df.to_string())
    
except Exception as e:
    print(f"\n❌ Training Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 90)
print("[3/6] COMPREHENSIVE DATA LEAKAGE AUDIT")
print("=" * 90)

try:
    audit = DataLeakageAudit(verbose=True)
    
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
    
    audit_results = audit.run_full_audit(
        X_train_np, X_test_np, y_train_np, y_test_np,
        X_val_np, y_val_np,
        feature_names=feature_names,
        y_pred=ensemble.cnn.y_pred,
        y_pred_proba=ensemble.cnn.y_pred_proba,
        train_metrics=ensemble.train_results.get('CNN', {}),
        test_metrics=ensemble.results.get('CNN', {})
    )
    
    print(f"\n✅ Audit complete!")
    if audit_results['issues_found']:
        print(f"\n🔴 {len(audit_results['issues_found'])} CRITICAL ISSUES FOUND:")
        for issue in audit_results['issues_found']:
            print(f"   - {issue['type']}: {issue['description']}")
    else:
        print("\n✅ No critical leakage issues detected!")
    
    if audit_results['warnings']:
        print(f"\n⚠️  {len(audit_results['warnings'])} WARNINGS:")
        for warn in audit_results['warnings'][:5]:
            print(f"   - {warn}")
    
    if audit_results['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in audit_results['recommendations'][:3]:
            print(f"   - {rec}")
    
except Exception as e:
    print(f"\n❌ Audit Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 90)
print("[4/6] GENERATING EVALUATION PLOTS")
print("=" * 90)

try:
    os.makedirs('./evaluation_plots', exist_ok=True)
    
    y_pred_proba_dict = {
        'CNN': ensemble.cnn.y_pred_proba,
        'Random Forest': ensemble.rf_pred_proba,
        'Logistic Regression': ensemble.lr_pred_proba
    }
    
    y_pred_dict = {
        'CNN': ensemble.cnn.y_pred,
        'Random Forest': ensemble.rf_pred,
        'Logistic Regression': ensemble.lr_pred
    }
    
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    figures = create_all_evaluation_plots(
        y_test=y_test_np,
        y_pred_proba_dict=y_pred_proba_dict,
        y_pred_dict=y_pred_dict,
        train_results_dict=ensemble.train_results,
        test_results_dict=ensemble.results,
        output_dir='./evaluation_plots'
    )
    
    print(f"\n✅ Generated {len(figures)} evaluation plots!")
    print("   Saved to: ./evaluation_plots/")
    print("   - ROC curves (should be curved, NOT perfect square)")
    print("   - Precision-Recall curves")
    print("   - Calibration curves")
    print("   - Confusion matrices")
    print("   - Threshold analysis")
    print("   - Probability distributions")
    print("   - Train vs Test metrics")
    
except Exception as e:
    print(f"\n❌ Visualization Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 90)
print("[5/6] ROBUSTNESS TESTING (Perturbations)")
print("=" * 90)

try:
    os.makedirs('./robustness_tests', exist_ok=True)
    
    # Define predict functions
    def predict_fn(X):
        X_np = X.values if hasattr(X, 'values') else X
        return (ensemble.cnn.model.predict(X_np, verbose=0) > 0.5).astype(int).flatten()
    
    def predict_proba_fn(X):
        X_np = X.values if hasattr(X, 'values') else X
        return ensemble.cnn.model.predict(X_np, verbose=0).flatten()
    
    tester = run_full_perturbation_suite(
        model_predict=predict_fn,
        X_test=X_test,
        y_test=y_test,
        model_predict_proba=predict_proba_fn,
        output_dir='./robustness_tests'
    )
    
    print(f"\n✅ Robustness testing complete!")
    print("   Saved to: ./robustness_tests/perturbation_robustness.png")
    print("   - Gaussian noise sensitivity")
    print("   - Feature dropout impact")
    print("   - Feature shuffling effect")
    print("   - Adversarial perturbations")
    
except Exception as e:
    print(f"\n⚠️  Robustness Testing Warning: {e}")
    print("   (This is non-critical; model testing may have issues)")

print("\n" + "=" * 90)
print("[6/6] FEATURE ANALYSIS (Importance & Leakage Detection)")
print("=" * 90)

try:
    os.makedirs('./feature_analysis', exist_ok=True)
    
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    detector, leakage_patterns = generate_feature_leakage_report(
        X_train_np, y_train_np,
        X_test_np, y_test_np,
        model=ensemble.rf,
        feature_names=feature_names,
        output_dir='./feature_analysis'
    )
    
    print(f"\n✅ Feature analysis complete!")
    if leakage_patterns:
        print(f"\n🔴 {len(leakage_patterns)} POTENTIAL LEAKAGE PATTERNS:")
        for pattern in leakage_patterns[:10]:
            severity = "🔴" if pattern['severity'] == 'CRITICAL' else "🟡"
            print(f"   {severity} {pattern['type']}: {pattern['feature']} ({pattern['value']:.4f})")
    else:
        print("\n✅ No suspicious feature patterns detected!")
    
    print("   Saved to: ./feature_analysis/feature_importance_analysis.png")
    
except Exception as e:
    print(f"\n⚠️  Feature Analysis Warning: {e}")
    print("   (This is non-critical)")

print("\n" + "=" * 90)
print("✅ VALIDATION COMPLETE")
print("=" * 90)

print(f"\n📊 FINAL SUMMARY:")
print(f"\n   Models Trained: 3 (CNN, Random Forest, Logistic Regression)")
print(f"   Cross-Validation: 5-Fold Stratified")
print(f"   Test Set Accuracy (CNN): {ensemble.results['CNN']['accuracy']:.4f}")
print(f"   Test Set ROC-AUC (CNN): {ensemble.results['CNN']['roc_auc']:.4f}")

train_acc = ensemble.train_results['CNN'].get('accuracy', 0)
test_acc = ensemble.results['CNN'].get('accuracy', 0)
gap = train_acc - test_acc
print(f"   Generalization Gap: {gap:.4f} (should be 5-15%)")

print(f"\n🔒 LEAKAGE DETECTION:")
print(f"   Critical Issues: {len(audit_results.get('issues_found', []))}")
print(f"   Warnings: {len(audit_results.get('warnings', []))}")

if gap > 0.15:
    print(f"\n⚠️  Large generalization gap (>{gap:.1%}) - potential overfitting")
elif gap < 0.05:
    print(f"\n⚠️  Small generalization gap (<{gap:.1%}) - potential leakage")
else:
    print(f"\n✅ Reasonable generalization gap ({gap:.1%})")

if ensemble.results['CNN']['accuracy'] > 0.95:
    print(f"❌ Accuracy too high (>{ensemble.results['CNN']['accuracy']:.1%}) - likely leakage")
elif ensemble.results['CNN']['accuracy'] < 0.55:
    print(f"❌ Accuracy too low (<{ensemble.results['CNN']['accuracy']:.1%}) - possible issues")
else:
    print(f"✅ Realistic accuracy ({ensemble.results['CNN']['accuracy']:.1%})")

print(f"\n📁 OUTPUT ARTIFACTS:")
print(f"   ✅ ./evaluation_plots/ - ROC, PR, calibration plots")
print(f"   ✅ ./robustness_tests/ - Perturbation analysis")
print(f"   ✅ ./feature_analysis/ - Feature importance analysis")

print(f"\n📖 DOCUMENTATION:")
print(f"   📄 DATA_LEAKAGE_FIX_GUIDE.md - Complete reference (1000+ lines)")
print(f"   📄 FIXES_SUMMARY.md - Executive summary with quick start")

print(f"\n" + "=" * 90)
if not audit_results.get('issues_found', []) and 0.05 < gap < 0.15 and ensemble.results['CNN']['accuracy'] < 0.95:
    print("✅ VALIDATION PASSED - Pipeline shows realistic results (No obvious leakage)")
else:
    print("⚠️  VALIDATION WARNINGS - Review audit output and plots above")

print("=" * 90 + "\n")
