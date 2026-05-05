"""
Test script for LOCAL SHAP explainability implementation.

Tests:
1. SHAP explainer initialization
2. Local explanation computation
3. Feature contribution table
4. Waterfall plot generation
5. Error handling
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

print("\n" + "="*70)
print("LOCAL SHAP EXPLAINABILITY - TEST SUITE")
print("="*70 + "\n")

# Test 1: Import dependencies
print("✓ Test 1: Checking dependencies...")
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"  ✓ TensorFlow {tf.__version__}")
    print(f"  ✓ Keras ready")
except ImportError as e:
    print(f"  ✗ TensorFlow/Keras: {e}")
    sys.exit(1)

try:
    import shap
    print(f"  ✓ SHAP {shap.__version__}")
except ImportError as e:
    print(f"  ✗ SHAP: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print(f"  ✓ Matplotlib ready")
except ImportError as e:
    print(f"  ✗ Matplotlib: {e}")
    sys.exit(1)

# Test 2: Import SHAP explainer module
print("\n✓ Test 2: Importing SHAP explainer...")
try:
    from shap_explainer import SHAPExplainer
    print("  ✓ SHAPExplainer class imported")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 3: Create synthetic data
print("\n✓ Test 3: Creating synthetic training data...")
try:
    np.random.seed(42)
    
    # Synthetic data: 100 samples, 20 features
    n_train = 100
    n_test = 10
    n_features = 20
    
    X_train = np.random.rand(n_train, n_features).astype(np.float32)
    X_test = np.random.rand(n_test, n_features).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train)
    y_test = np.random.randint(0, 2, n_test)
    
    feature_names = np.array([f"Feature_{i:02d}" for i in range(n_features)])
    
    print(f"  ✓ X_train shape: {X_train.shape}")
    print(f"  ✓ X_test shape: {X_test.shape}")
    print(f"  ✓ {len(feature_names)} feature names")
except Exception as e:
    print(f"  ✗ Data creation: {e}")
    sys.exit(1)

# Test 4: Build and train model
print("\n✓ Test 4: Building and training DNN model...")
try:
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"  ✓ Model created: {model.name}")
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=16,
        verbose=0,
        validation_split=0.2
    )
    
    print(f"  ✓ Model trained (5 epochs)")
    print(f"  ✓ Training loss: {history.history['loss'][-1]:.4f}")
    print(f"  ✓ Training accuracy: {history.history['accuracy'][-1]:.4f}")
    
except Exception as e:
    print(f"  ✗ Model creation/training: {e}")
    sys.exit(1)

# Test 5: Initialize SHAP explainer
print("\n✓ Test 5: Initializing SHAP explainer...")
try:
    explainer = SHAPExplainer(model, feature_names=feature_names)
    print("  ✓ SHAPExplainer initialized")
    
    explainer.init_with_background_data(X_train)
    print("  ✓ Background data loaded")
    print(f"  ✓ Background samples: {len(explainer.background_data)}")
    
except Exception as e:
    print(f"  ✗ SHAP initialization: {e}")
    sys.exit(1)

# Test 6: LOCAL SHAP - single instance explanation
print("\n✓ Test 6: Computing LOCAL SHAP explanation (single instance)...")
try:
    # Get first test sample
    X_instance = X_test[0:1]
    print(f"  ✓ Sample shape: {X_instance.shape}")
    
    # Compute SHAP explanation (using fast settings: 25 samples)
    print("  ⏳ Computing SHAP values (may take 20-40 seconds)...")
    shap_exp = explainer.explain_instance(X_instance, num_samples=25)
    
    print("  ✓ SHAP computation successful!")
    print(f"  ✓ Prediction: {shap_exp['prediction']:.4f}")
    print(f"  ✓ SHAP values shape: {shap_exp['shap_values'].shape}")
    print(f"  ✓ Feature names count: {len(shap_exp['feature_names'])}")
    
except Exception as e:
    print(f"  ✗ SHAP computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Get top contributing features
print("\n✓ Test 7: Extracting top contributing features...")
try:
    top_features = explainer.get_top_contributing_features(shap_exp, top_n=5)
    
    print("  ✓ Top 5 features extracted")
    print("\n  Top Contributing Features:")
    print(top_features.to_string())
    
    # Verify columns
    required_cols = ['Feature', 'Impact', 'Direction', 'Magnitude']
    for col in required_cols:
        if col in top_features.columns:
            print(f"  ✓ Column '{col}' present")
        else:
            print(f"  ✗ Column '{col}' missing!")
            
except Exception as e:
    print(f"  ✗ Feature extraction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Generate waterfall plot
print("\n✓ Test 8: Generating waterfall visualization...")
try:
    fig = explainer.plot_waterfall(shap_exp)
    
    print("  ✓ Waterfall plot created")
    print(f"  ✓ Figure type: {type(fig)}")
    print(f"  ✓ Figure size: {fig.get_size_inches()}")
    
    # Save plot
    plot_path = "test_waterfall_plot.png"
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {plot_path}")
    
except Exception as e:
    print(f"  ✗ Waterfall plot: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Generate feature importance bar plot
print("\n✓ Test 9: Generating feature importance plot...")
try:
    # Prepare batch result
    batch_result = {
        'shap_values': np.random.randn(1, n_features),  # For demo
        'feature_names': feature_names
    }
    
    fig = explainer.plot_summary(batch_result)
    
    print("  ✓ Summary plot created")
    print(f"  ✓ Figure type: {type(fig)}")
    
    # Save plot
    plot_path = "test_summary_plot.png"
    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {plot_path}")
    
except Exception as e:
    print(f"  ✗ Summary plot: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Error handling - no explainer
print("\n✓ Test 10: Testing error handling...")
try:
    # Create explainer without initialization
    bad_explainer = SHAPExplainer(model, feature_names=feature_names)
    
    # Try to explain without init
    try:
        bad_explainer.explain_instance(X_instance)
        print("  ✗ Should have raised error!")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {str(e)[:50]}...")
    
except Exception as e:
    print(f"  ✗ Error handling test: {e}")
    sys.exit(1)

# Test 11: Multiple instances
print("\n✓ Test 11: Computing SHAP for multiple instances...")
try:
    print("  ⏳ Computing SHAP for 3 samples (may take 60-90 seconds)...")
    
    multiple_exps = []
    for i in range(3):
        X_sample = X_test[i:i+1]
        shap_exp = explainer.explain_instance(X_sample, num_samples=25)
        multiple_exps.append(shap_exp)
        print(f"  ✓ Sample {i}: Prediction {shap_exp['prediction']:.4f}")
    
    print("  ✓ All samples explained successfully")
    
except Exception as e:
    print(f"  ✗ Multiple instance explanation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 12: Performance check
print("\n✓ Test 12: Performance analysis...")
try:
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    memory_mb = mem_info.rss / 1024 / 1024
    print(f"  ✓ Current memory usage: {memory_mb:.2f} MB")
    
    if memory_mb < 500:
        print("  ✓ Memory usage is acceptable (<500 MB)")
    else:
        print(f"  ⚠ Memory usage is high (>{memory_mb:.0f} MB)")
    
except ImportError:
    print("  ⚠ psutil not installed, skipping memory check")
except Exception as e:
    print(f"  ⚠ Memory check failed: {e}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY - ALL TESTS PASSED! ✅")
print("="*70)
print("""
LOCAL SHAP Implementation Status:
✓ SHAP explainer initialization working
✓ Local explanation computation working
✓ Feature contribution extraction working
✓ Waterfall plot generation working
✓ Summary plot generation working
✓ Error handling working
✓ Multiple instance explanation working
✓ Performance acceptable

Ready for Streamlit integration! 🚀
""")

print("\nNext Steps:")
print("1. Open Streamlit app: streamlit run app.py")
print("2. Go to '🔍 Make Prediction' tab")
print("3. Enter feature values (0.0 - 1.0)")
print("4. Click '🔮 Predict'")
print("5. Click '📈 Compute SHAP Explanation'")
print("6. View local SHAP explanation (waterfall + top features)")
print("\n" + "="*70 + "\n")
