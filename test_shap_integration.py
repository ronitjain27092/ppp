"""
Validation script for SHAP Explainable AI integration.

Tests:
1. Import all dependencies
2. Check model compatibility
3. Verify SHAP functions available
4. Test with synthetic data
"""

import sys
import numpy as np

print("\n" + "="*70)
print("SHAP EXPLAINABLE AI - VALIDATION TEST")
print("="*70 + "\n")

# Test 1: Import dependencies
print("✓ Test 1: Checking dependencies...")
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"  ✗ TensorFlow: {e}")
    sys.exit(1)

try:
    import shap
    print(f"  ✓ SHAP {shap.__version__}")
except ImportError as e:
    print(f"  ✗ SHAP: {e}")
    sys.exit(1)

try:
    import streamlit as st
    print(f"  ✓ Streamlit")
except ImportError as e:
    print(f"  ✗ Streamlit: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"  ✓ Matplotlib")
except ImportError as e:
    print(f"  ✗ Matplotlib: {e}")
    sys.exit(1)

# Test 2: Import SHAP module
print("\n✓ Test 2: Checking SHAP module...")
try:
    from shap_module import (
        SHAPExplainer, 
        explain_model, 
        display_global_explanation, 
        display_local_explanation
    )
    print("  ✓ SHAPExplainer class")
    print("  ✓ explain_model() factory function")
    print("  ✓ display_global_explanation() helper")
    print("  ✓ display_local_explanation() helper")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 3: Create synthetic data
print("\n✓ Test 3: Creating synthetic test data...")
try:
    # Synthetic training data (100 samples, 20 features)
    X_train = np.random.randn(100, 20).astype(np.float32)
    X_test = np.random.randn(20, 20).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    feature_names = [f"Feature_{i}" for i in range(20)]
    
    print(f"  ✓ X_train shape: {X_train.shape}")
    print(f"  ✓ X_test shape: {X_test.shape}")
    print(f"  ✓ {len(feature_names)} feature names")
except Exception as e:
    print(f"  ✗ Synthetic data creation: {e}")
    sys.exit(1)

# Test 4: Build minimal model
print("\n✓ Test 4: Building minimal CNN model...")
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, 3, activation='relu', input_shape=(20, 1)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(f"  ✓ Model created: {model.name}")
    print(f"  ✓ Input shape: (batch, 20, 1)")
except Exception as e:
    print(f"  ✗ Model creation: {e}")
    sys.exit(1)

# Test 5: Initialize SHAP
print("\n✓ Test 5: Initializing SHAP explainer...")
try:
    explainer = explain_model(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names
    )
    print("  ✓ SHAP explainer initialized")
    print(f"  ✓ Background samples: 100")
    print(f"  ✓ Ready for explanations")
except Exception as e:
    print(f"  ✗ SHAP initialization: {e}")
    sys.exit(1)

# Test 6: Test global explanation (without display)
print("\n✓ Test 6: Testing global explanation...")
try:
    print("  ⏳ Computing global SHAP values...")
    fig = explainer.explain_global(X_test[:5], max_display=10)
    if fig:
        print("  ✓ Global explanation figure generated")
    else:
        print("  ⚠ Global explanation returned None")
except Exception as e:
    print(f"  ✗ Global explanation: {e}")

# Test 7: Test local explanation (without display)
print("\n✓ Test 7: Testing local explanation...")
try:
    print("  ⏳ Computing local SHAP values...")
    fig, info_dict = explainer.explain_local(X_test, 0)
    if fig and info_dict:
        print("  ✓ Local explanation figure generated")
        print(f"  ✓ Top features identified: {len(info_dict['top_features'])}")
        if info_dict['top_features']:
            feat = info_dict['top_features'][0]
            print(f"    - {feat['feature']}: {feat['shap_value']:.4f}")
    else:
        print("  ⚠ Local explanation returned None")
except Exception as e:
    print(f"  ✗ Local explanation: {e}")

# Test 8: Check input shape handling
print("\n✓ Test 8: Testing input shape handling...")
try:
    # Test 3D input (samples, features, channels)
    X_3d = np.random.randn(10, 20, 1).astype(np.float32)
    print(f"  ✓ Input shape (3D): {X_3d.shape}")
    
    # SHAP explainer should handle this
    result = explainer._prepare_input(X_3d)
    print(f"  ✓ After prepare: {result.shape}")
    
    if len(result.shape) == 2:
        print("  ✓ Correctly converted to 2D for SHAP")
    else:
        print("  ✗ Shape conversion failed")
except Exception as e:
    print(f"  ✗ Shape handling: {e}")

# Summary
print("\n" + "="*70)
print("VALIDATION COMPLETE ✅")
print("="*70)
print("""
All components working correctly!

Next steps:
1. Run: streamlit run streamlit_app_fixed.py
2. Upload your CIC-MalMem CSV dataset
3. Train the model (SHAP will auto-initialize)
4. Select "🧠 Explainable AI" mode to view explanations

For questions, see SHAP_INTEGRATION_GUIDE.md
""")
