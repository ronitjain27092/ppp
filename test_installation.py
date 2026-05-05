"""
Test Script - Verify installation and create sample data
This script:
1. Tests all imports
2. Creates sample dataset
3. Tests preprocessing
4. Trains a small model
5. Tests SHAP explanations
"""

import sys
import os
import numpy as np
import pandas as pd

print("\n" + "="*70)
print("TESTING MALWARE DETECTION XAI PROJECT")
print("="*70)

# Test 1: Check Python version
print("\n[1/6] Checking Python version...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 8:
    print(f"✓ Python {python_version.major}.{python_version.minor} OK")
else:
    print(f"✗ Python {python_version.major}.{python_version.minor} (Need 3.8+)")
    sys.exit(1)

# Test 2: Test imports
print("\n[2/6] Testing imports...")
try:
    import tensorflow as tf
    print("✓ TensorFlow imported")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import shap
    print("✓ SHAP imported")
except ImportError as e:
    print(f"✗ SHAP import failed: {e}")
    sys.exit(1)

try:
    import streamlit as st
    print("✓ Streamlit imported")
except ImportError as e:
    print(f"✗ Streamlit import failed: {e}")
    sys.exit(1)

try:
    from preprocessing import DataPreprocessor
    from model import MalwareDetectionModel
    from explain import SHAPExplainer
    print("✓ Custom modules imported")
except ImportError as e:
    print(f"✗ Custom module import failed: {e}")
    sys.exit(1)

# Test 3: Create sample dataset
print("\n[3/6] Creating sample dataset...")
np.random.seed(42)

# Generate synthetic malware dataset
n_samples = 200
n_features = 20

X_benign = np.random.normal(loc=0.3, scale=0.15, size=(100, n_features))
X_malware = np.random.normal(loc=0.7, scale=0.15, size=(100, n_features))

X = np.vstack([X_benign, X_malware])
y = np.hstack([np.zeros(100), np.ones(100)])

# Ensure values are in [0, 1]
X = np.clip(X, 0, 1)

# Create feature names
feature_names = [f"Feature_{i:02d}" for i in range(n_features)]

# Create dataframe
df = pd.DataFrame(X, columns=feature_names)
df['Class'] = y
df['Class'] = df['Class'].astype(int).map({0: 'Benign', 1: 'Malware'})

# Save dataset
sample_csv = "sample_dataset.csv"
df.to_csv(sample_csv, index=False)
print(f"✓ Sample dataset created: {sample_csv}")
print(f"  - {len(df)} samples, {len(feature_names)} features")
print(f"  - Benign: {(df['Class']=='Benign').sum()}, Malware: {(df['Class']=='Malware').sum()}")

# Test 4: Test preprocessing
print("\n[4/6] Testing preprocessing...")
try:
    preprocessor = DataPreprocessor()
    X_processed, y_processed, feat_names, df_processed = preprocessor.preprocess(sample_csv)
    print(f"✓ Preprocessing completed")
    print(f"  - Processed shape: {X_processed.shape}")
    print(f"  - Features detected: {len(feat_names)}")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    sys.exit(1)

# Test 5: Test model training
print("\n[5/6] Testing model training...")
try:
    model = MalwareDetectionModel(model_type='dnn')
    X_train, X_test, y_train, y_test = model.split_data(X_processed, y_processed, test_size=0.2)
    model.create_model(X_train.shape[1])
    print("✓ Model created")
    
    # Train briefly
    print("  Training for 5 epochs...")
    model.train(epochs=5, batch_size=16, verbose=0)
    print("✓ Model training completed")
    print(f"  - Accuracy: {model.metrics['accuracy']:.4f}")
    print(f"  - Precision: {model.metrics['precision']:.4f}")
    print(f"  - Recall: {model.metrics['recall']:.4f}")
    print(f"  - F1-Score: {model.metrics['f1']:.4f}")
except Exception as e:
    print(f"✗ Model training failed: {e}")
    sys.exit(1)

# Test 6: Test SHAP explainer
print("\n[6/6] Testing SHAP explainer...")
try:
    explainer = SHAPExplainer(model.model, feat_names)
    print("✓ SHAP explainer created")
    
    explainer.create_explainer(X_train, sample_size=50)
    print("✓ SHAP explainer initialized")
    
    # Get feature importance
    importance = explainer.get_feature_importance(X_test, num_samples=20)
    print("✓ Feature importance calculated")
    print(f"  - Top feature: {importance.iloc[0]['Feature']}")
except Exception as e:
    print(f"✗ SHAP explainer failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)

print("\nProject is ready to use!")
print("\nNext steps:")
print("1. Replace 'sample_dataset.csv' with your CIC-MalMem-2022 dataset")
print("2. Run: streamlit run app.py")
print("3. Upload your dataset and train the model")
print("\nOr test with sample data:")
print("  streamlit run app.py")
print("  Go to '📊 Train Model' tab")
print("  Upload: sample_dataset.csv")

# Clean up (optional - comment out to keep sample data)
# os.remove(sample_csv)

print("\n" + "="*70)
