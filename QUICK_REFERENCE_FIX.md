# Quick Reference: Class Imbalance Fix

## TL;DR (Too Long; Didn't Read)

**Problem:** Model says 99% accurate but detects 0% of malware.
**Fix:** Apply class weights + lower threshold from 0.5 to 0.3.
**Result:** Now detects both Benign and Malware properly.

---

## One-Minute Summary

| Aspect | Details |
|--------|---------|
| **What's Wrong** | Dataset is 90% Benign, 10% Malware → Model just predicts Benign always |
| **Why It's Wrong** | No class weights (all errors treated equal) + threshold 0.5 too strict |
| **How To Fix** | Use `class_weight='balanced'` + threshold 0.3 |
| **Files** | `cnn_model_fixed.py`, `evaluation_fixed.py`, `streamlit_app_fixed.py` |
| **To Run** | `streamlit run streamlit_app_fixed.py` |

---

## The Fix in 3 Steps

### Step 1: Apply Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=[0,1], y=y_train)
# Result: {0: 0.556, 1: 5.0}
# Meaning: Malware misclassification costs 5x as much!

model.fit(X_train, y_train, class_weight=class_weights)
```

### Step 2: Lower Threshold
```python
# Default (WRONG):
y_pred = (y_pred_proba >= 0.5).astype(int)

# Fixed (CORRECT):
y_pred = (y_pred_proba >= 0.3).astype(int)
```

### Step 3: Better Architecture
```python
# Use CNN1D with regularization
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(50, 1)),
    BatchNormalization(),
    Dropout(0.3),
    # ... more layers ...
    Dense(1, activation='sigmoid')
])
```

---

## Check If Fix Is Working

Run this after training:
```python
from evaluation_fixed import ModelEvaluator

evaluator = ModelEvaluator(model, y_test)
evaluator.analyse_predictions()
```

✅ **Success if you see:**
- ✓ Recall > 0.75 (catching 75%+ of malware)
- ✓ F1-Score > 0.70 (good balance)
- ✓ Both Benign AND Malware in confusion matrix
- ✓ No alerts about "Recall = 0"

❌ **Problem if you see:**
- ✗ Recall = 0 (still only detecting Benign)
- ✗ F1-Score = 0
- ✗ All predictions same class
- ✗ Alert: "Consider lowering threshold further"

---

## Metrics Reference

### Bad (Imbalanced - Before Fix):
```
Accuracy:  99%  ✗ Misleading
Precision: 0%   ✗ No true positives
Recall:    0%   ✗ Not detecting malware
F1-Score:  0%   ✗ Useless
```

### Good (Balanced - After Fix):
```
Accuracy:  85-92%  ✓ Realistic
Precision: 70-85%  ✓ Good
Recall:    75-90%  ✓ Catches malware
F1-Score:  72-87%  ✓ Good balance
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Recall still 0%" | Lower threshold to 0.2 and retry |
| "Too many false alarms" | Raise threshold to 0.4 (trade-off) |
| "Model only predicts Benign" | Check class weights computed correctly |
| "NoneType error" | Make sure model trained before evaluation |
| "Different results each run" | Set random seeds (TensorFlow, NumPy) |

---

## File Guide

### Use This File | To Do This
|---|---|
| `cnn_model_fixed.py` | Train model with class weights |
| `evaluation_fixed.py` | Check metrics & find optimal threshold |
| `streamlit_app_fixed.py` | Run web UI (easiest!) |
| `CLASS_IMBALANCE_FIX_GUIDE.md` | Deep understanding of the problem |
| `PHASE_4_COMPLETION.md` | See all changes made |

---

## Code Reference

### Building Model with Class Weights:
```python
from cnn_model_fixed import CNNMalwareDetector

model = CNNMalwareDetector(input_shape=(50,))
model.compute_class_weights(y_train)
model.train(X_train, y_train, epochs=20, batch_size=32)
```

### Evaluating with Threshold Adjustment:
```python
from evaluation_fixed import ModelEvaluator

evaluator = ModelEvaluator(model, y_test)
evaluator.analyse_predictions()  # Shows comprehensive analysis

# Find best threshold
best_threshold = evaluator.find_optimal_threshold(metric='f1')
print(f"Recommended threshold: {best_threshold}")
```

### Running Web App:
```bash
streamlit run streamlit_app_fixed.py
```
Then:
1. Click "📊 Train Model"
2. Upload CSV files
3. See class weights and metrics
4. Click "📚 About the Fix" to learn

---

## What Changed?

| Component | Before | After |
|-----------|--------|-------|
| Class Weights | None (all equal) | Balanced (Malware 5-9x heavier) |
| Threshold | 0.5 (fixed) | 0.3 (adjusted) |
| Model | Basic DL | CNN1D + BatchNorm + Dropout |
| Evaluation | Accuracy only | Recall, Precision, F1, Confusion Matrix |
| Debug output | Minimal | Comprehensive with alerts |

---

## Why It Works

```
Before:       After:
Benign ████  Benign ██
Malware ▌    Malware ████
           ↑ More weight on Malware!
```

- Class weights: "Penalize if you're wrong about Malware"
- Lower threshold: "Predict Malware if even 30% confident"
- Better architecture: "Learn patterns, not just 'always Benign'"

---

## Performance Expectations

With 10,000 samples (9000 Benign, 1000 Malware):

| Metric | Achievable | How To Verify |
|--------|-----------|---------------|
| Recall | 75-90% | `ModelEvaluator.analyse_predictions()` |
| Precision | 70-85% | Same |
| F1-Score | 72-87% | Same |
| Confusion | Both classes | Plot shows TP, FP, FN, TN |

If not achieving these → Try:
1. Lower threshold further (0.2)
2. Check class distribution (should be imbalanced)
3. Increase training epochs
4. Use more data

---

## Quick Test

```python
from cnn_model_fixed import CNNMalwareDetector
from evaluation_fixed import ModelEvaluator

# 1. Build model
model = CNNMalwareDetector(input_shape=(X_train.shape[1],))

# 2. Compute class weights  
model.compute_class_weights(y_train)
print("✓ Class weights computed")

# 3. Train
model.train(X_train, y_train, epochs=15)
print("✓ Model trained")

# 4. Evaluate
evaluator = ModelEvaluator(model, y_test)
evaluator.analyse_predictions()
print("✓ Evaluation complete")

# 5. Check success
recall = evaluator.recalled.recall_score()
if recall > 0.75:
    print("✅ SUCCESS: Recall > 75%")
else:
    print("❌ FAILED: Adjust threshold or retrain")
```

---

## Key Numbers

- **Class weight ratio**: 5-9x (Malware vs Benign)
- **Threshold**: 0.3 (not 0.5)
- **Dropout**: 30-40%
- **Early stop patience**: 3 epochs
- **Expected recall**: > 75%
- **Expected F1**: > 0.70

---

## Support

📖 **For Understanding:** Read `CLASS_IMBALANCE_FIX_GUIDE.md`
🚀 **For Implementation:** Use `streamlit_app_fixed.py`
🔍 **For Debugging:** Check `evaluation_fixed.py` output
❓ **For Questions:** Check Common Issues table above

---

## Success Criteria

✅ All of these must be true:

```
1. Class weights printed during training
2. Recall >= 0.75 (at least 75% of malware detected)
3. F1-Score > 0.70 (good balance)
4. Confusion matrix shows BOTH classes
5. No "Recall = 0" alerts
6. No "Only one class predicted" warnings
7. Model didn't crash
8. Threshold = 0.3 (not 0.5)
```

If any fail → Refer to "Common Issues & Solutions" table.

---

**This is it! You have everything needed to fix the class imbalance issue.** 🎉
