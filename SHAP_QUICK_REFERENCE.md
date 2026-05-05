# SHAP Quick Reference - Malware Detection XAI

## 🎯 Quick Start

### 1. Initialize SHAP After Training
```python
from shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model, feature_names)
explainer.init_with_background_data(X_train_scaled)
```

### 2. Get Global Explanations
```python
shap_result = explainer.explain_batch(X_test, max_samples=50)
importance_df = explainer.get_feature_importance(shap_result, top_n=10)
fig = explainer.plot_summary(shap_result)
```

### 3. Get Local Explanations
```python
shap_exp = explainer.explain_instance(X_sample)
top_features_df = explainer.get_top_contributing_features(shap_exp, top_n=5)
fig = explainer.plot_waterfall(shap_exp)
```

---

## 📚 API Reference

### SHAPExplainer Class

#### Constructor
```python
SHAPExplainer(model, feature_names=None)
```
- `model`: Trained Keras/TensorFlow model
- `feature_names`: List of feature names (optional)

#### Methods

| Method | Purpose | Returns | Time |
|--------|---------|---------|------|
| `init_with_background_data(X)` | Initialize with background data | None | ~2s |
| `explain_instance(X)` | Single sample explanation | dict | ~2-5s |
| `explain_batch(X, max_samples=50)` | Multiple sample explanation | dict | ~5-10s |
| `get_feature_importance(result, top_n)` | Feature importance ranking | DataFrame | <1s |
| `plot_summary(result)` | Feature importance bar plot | Figure | ~1-2s |
| `plot_waterfall(result)` | Cumulative feature plot | Figure | ~1-2s |
| `get_top_contributing_features(result, top_n)` | Top features for prediction | DataFrame | <1s |

---

## 🔍 Understanding SHAP Values

### SHAP Value = Feature Contribution to Prediction

```
SHAP Value > 0  →  🔴 Feature pushes toward MALWARE
SHAP Value < 0  →  🟢 Feature pushes toward BENIGN
|SHAP Value|    →  Magnitude of importance
```

### Example: Predicting Malware

```
Base Value (model default):           0.35
├─ SyscallCount=250    (SHAP: +0.25) → 0.60
├─ ProcessNameLen=45   (SHAP: +0.15) → 0.75
├─ MemoryUsage=512     (SHAP: -0.05) → 0.70
└─ IOReadCount=2048    (SHAP: +0.20) → 0.90 ← MALWARE

Interpretation:
- SyscallCount and IOReadCount are most suspicious
- More syscalls + high IO = likely malware
```

---

## 📊 SHAP Visualizations

### Global Explanation - Feature Importance Bar Plot
```
ProcessNameLen   ███████████ (importance: 0.45)
SyscallCount     ██████      (importance: 0.28)
ThreadCount      ████        (importance: 0.18)
IOReadCount      ███         (importance: 0.12)
```

**What it shows**: Which features matter most overall

### Local Explanation - Waterfall Plot
```
Base Value (0.35)
  ├─ +SyscallCount     (0.15) ↗
  ├─ +ProcessNameLen   (0.08) ↗
  ├─ +IOReadCount      (0.20) ↗
  └─ -MemoryUsage      (-0.03) ↙
     Final (0.85) → MALWARE
```

**What it shows**: How each feature contributed to this specific prediction

---

## ⚙️ Configuration

### Background Data
```python
# Automatic: Uses first 100 training samples
explainer.init_with_background_data(X_train_scaled)

# Why 100?
# - Too few (<50):   Inaccurate explanations
# - Just right (100): Balance of speed & accuracy
# - Too many (>500): Slow computation
```

### Batch Size for Global Explanations
```python
# Analyze 50 test samples (recommended)
shap_result = explainer.explain_batch(X_test, max_samples=50)

# Smaller = faster but less representative
# Larger = slower but more reliable
```

### Top Features
```python
# Show top 10 features for importance
importance_df = explainer.get_feature_importance(result, top_n=10)

# Show top 5 for single prediction
top_features = explainer.get_top_contributing_features(result, top_n=5)
```

---

## 🐛 Error Handling

### Common Issues

**Issue**: `ValueError: Explainer not initialized`
```python
# Solution: Call init_with_background_data() first
explainer.init_with_background_data(X_train_scaled)
```

**Issue**: `TypeError: X_batch shape mismatch`
```python
# Ensure 2D array (n_samples, n_features)
X = np.array(X).reshape(-1, n_features)
```

**Issue**: Features missing in plot
```python
# Ensure feature_names passed correctly
explainer = SHAPExplainer(model, feature_names=['Feat1', 'Feat2', ...])
```

---

## 📈 Performance Tips

### Speed Up Global Explanations
```python
# Reduce batch size
shap_result = explainer.explain_batch(X_test, max_samples=20)  # Faster

# Reduce background data
X_bg = X_train_scaled[:50]  # Smaller background
explainer.init_with_background_data(X_bg)
```

### Speed Up Local Explanations
```python
# Usually already fast (~2-5s)
# Limited by DeepExplainer gradient computation
# Can't optimize further without changing method
```

### Cache Results
```python
# Store SHAP results to avoid recomputation
st.session_state['last_shap_result'] = shap_result

# Reuse if same data
if np.array_equal(X_new, X_old):
    shap_result = st.session_state['last_shap_result']
else:
    shap_result = explainer.explain_batch(X_new)
```

---

## 🎨 Customization

### Custom Feature Names
```python
custom_names = {
    'Feat1': 'System Calls Count',
    'Feat2': 'Process Name Length',
    'Feat3': 'Memory Usage',
}

feature_names = [custom_names.get(f, f) for f in original_features]
explainer = SHAPExplainer(model, feature_names)
```

### Custom Thresholds
```python
# SHAP works with any threshold
# Default in app: 0.5 (50% probability)

# Change prediction threshold
prediction_prob = model.predict(X)[0][0]
if prediction_prob > 0.3:  # Custom threshold
    predicted_class = "MALWARE (HIGH CONFIDENCE)"
elif prediction_prob > 0.5:
    predicted_class = "MALWARE (MEDIUM CONFIDENCE)"
```

### Custom Plot Styling
```python
import matplotlib.pyplot as plt

fig = explainer.plot_summary(shap_result)
fig.set_size_inches(14, 8)
plt.title("Custom Title", fontsize=16)
plt.tight_layout()
```

---

## 📝 Common Patterns

### Pattern 1: Explain Model Decision
```python
# Train model
model = train_model(X_train, y_train)

# Initialize explainer
explainer = SHAPExplainer(model, feature_names)
explainer.init_with_background_data(X_train)

# Get explanation
shap_exp = explainer.explain_instance(X_sample)

# Interpret
prediction = shap_exp['prediction']
top_features = explainer.get_top_contributing_features(shap_exp)
print(f"Predicted: {prediction:.2f} (malware)")
print("Top contributing features:")
print(top_features)
```

### Pattern 2: Compare Feature Importance
```python
# Get explanations for different models/datasets
result1 = explainer1.explain_batch(X_test1)
result2 = explainer2.explain_batch(X_test2)

# Compare
imp1 = explainer1.get_feature_importance(result1, top_n=10)
imp2 = explainer2.get_feature_importance(result2, top_n=10)

# Find differences
diff = set(imp1['Feature']) - set(imp2['Feature'])
print(f"Model 1 unique features: {diff}")
```

### Pattern 3: Batch Prediction Explanation
```python
# Explain multiple predictions
predictions = []
for X_sample in X_batch:
    shap_exp = explainer.explain_instance(X_sample)
    predictions.append({
        'sample': X_sample,
        'prediction': shap_exp['prediction'],
        'top_features': explainer.get_top_contributing_features(shap_exp)
    })

# Save results
results_df = pd.DataFrame([
    {
        'sample_id': i,
        'prediction': p['prediction'],
        'top_feature': p['top_features'].iloc[0]['Feature']
    }
    for i, p in enumerate(predictions)
])
```

---

## 🌍 DeepExplainer vs KernelExplainer

### Use DeepExplainer (Current)
✅ Fast (gradient-based)
✅ TensorFlow/Keras native
✅ Good for neural networks
❌ Approximation (less precise)

### Use KernelExplainer (Fallback)
✅ Model-agnostic
✅ Highly accurate
✅ Any model type
❌ Slow (1000+ evaluations)

**Current**: Uses DeepExplainer, falls back to Kernel if needed

---

## 📚 Further Reading

**SHAP Paper**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- https://arxiv.org/abs/1705.07874

**Official Docs**: https://shap.readthedocs.io/

**Interactive Examples**: https://github.com/slundberg/shap

---

## 🏆 SHAP Values Properties

### 1. Local Accuracy
SHAP values sum to the difference between prediction and base value
```
sum(shap_values) + base_value = final_prediction
```

### 2. Missingness
If feature is missing, SHAP value is based on expected value

### 3. Consistency
If model output increases, feature's SHAP value increases or stays same

### 4. Uniqueness
Each feature gets exactly the right credit/blame (Shapley fairness)

---

## 💡 Tips for Interpretation

1. **Check base value**: What's model's default prediction?
2. **Look at magnitudes**: Largest SHAP values matter most
3. **Check signs**: Positive = malware indicator, Negative = benign indicator
4. **Verify with domain knowledge**: Do features match security threat intel?
5. **Look for patterns**: Similar samples should have similar SHAP explanations
6. **Check for bias**: Are certain populations over/under represented?

---

**Last Updated**: Current Session
**Version**: 1.0 - SHAP Integration Complete
