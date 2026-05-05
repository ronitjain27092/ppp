# LOCAL SHAP - Quick Reference Card

## ⚡ 30-Second Summary

**What:** Explain each malware prediction with SHAP  
**How:** Click "📈 Compute SHAP Explanation" button  
**Time:** 30-60 seconds per prediction  
**Output:** Top 5 features + waterfall visualization  
**Status:** ✅ Fully implemented and tested  

---

## 🎯 User Quick Start

### 1. Train Model
```
📊 Train Model tab → Upload CSV → "🚀 START TRAINING"
```

### 2. Make Prediction
```
🔍 Make Prediction tab → Enter values (0.0-1.0) → "🔮 Predict"
```

### 3. Get Explanation
```
📈 "Compute SHAP Explanation" → Wait 30-60 sec → View results
```

### 4. Interpret Results
```
🔴 Red bar = MALWARE indicator
🟢 Blue bar = BENIGN indicator
Bar length = Impact strength
```

---

## 📊 What You'll See

### Table (Top 5 Features)
```
Feature          SHAP Value  Direction     Importance
────────────────────────────────────────────────────
Entropy          +0.35       🔴 Malware   0.35
API_Calls        +0.22       🔴 Malware   0.22
File_Size        -0.12       🟢 Benign    0.12
Permissions      +0.08       🔴 Malware   0.08
Syscalls         -0.05       🟢 Benign    0.05
```

### Waterfall Plot
```
Entropy      ██████████████████████████ 0.35
API_Calls    ███████████████ 0.22
File_Size    ███████ (negative)
Permissions  █████ 0.08
Syscalls     ███ (negative)
```

---

## 🔴 Red vs 🟢 Blue

| Color | Feature | Effect |
|-------|---------|--------|
| 🔴 Red | Suspicious API call | → Suggests MALWARE |
| 🔴 Red | High entropy | → Suggests MALWARE |
| 🟢 Blue | Normal file size | → Suggests BENIGN |
| 🟢 Blue | Standard permissions | → Suggests BENIGN |

---

## ⏱️ Expected Times

| Operation | Time |
|-----------|------|
| Model training | 2-5 minutes |
| Prediction | <1 second |
| SHAP explanation | 30-60 seconds |
| Waterfall plot | 1-2 seconds |

---

## 🛠️ Methods (For Developers)

### Compute Local Explanation
```python
shap_exp = explainer.explain_instance(X_sample, num_samples=50)
# Returns: dict with shap_values, prediction, etc.
# Time: 30-60 seconds
```

### Fast Mode
```python
shap_exp = explainer.explain_instance_fast(X_sample)
# Uses num_samples=25
# Time: 15-30 seconds
```

### Get Top Features
```python
top_5 = explainer.get_top_contributing_features(shap_exp, top_n=5)
# Returns: DataFrame with Feature, Impact, Direction, Magnitude
# Time: <1 second
```

### Plot Waterfall
```python
fig = explainer.plot_waterfall(shap_exp)
st.pyplot(fig)
# Displays horizontal bar chart
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app (Make Prediction tab, lines 474-527) |
| `shap_explainer.py` | SHAP explainer class with explain_instance() |
| `test_local_shap.py` | Test suite (12 tests, all passing) |
| `LOCAL_SHAP_IMPLEMENTATION_GUIDE.md` | Full technical docs |
| `LOCAL_SHAP_QUICK_START.md` | Detailed quick start |
| `LOCAL_SHAP_FINAL_SUMMARY.md` | Implementation summary |

---

## 🐛 Troubleshooting

### "SHAP explainer not initialized"
✅ **Fix:** Train model in "📊 Train Model" tab first

### "SHAP computation failed: timeout"
✅ **Fix:** Try again, or use `explain_instance_fast()`

### "Per-column arrays must each be 1-dimensional"
✅ **Fix:** Already fixed! Update to latest version

### Feature values not 0-1 range
✅ **Fix:** Normalize to [0.0, 1.0] range before entering

---

## 💻 Code Examples

### Single Prediction Explanation
```python
from shap_explainer import SHAPExplainer

# Initialize
explainer = SHAPExplainer(model, feature_names)
explainer.init_with_background_data(X_train)

# Explain prediction
X_test = np.array([[0.5, 0.8, 0.3, ...]])  # 1 sample
shap_exp = explainer.explain_instance(X_test, num_samples=50)

# View results
print(f"Prediction: {shap_exp['prediction']:.2%}")
print(f"Top feature: {shap_exp['feature_names'][0]}")
```

### Get Top Features
```python
top_features = explainer.get_top_contributing_features(
    shap_exp, 
    top_n=5
)

print(top_features)
#     Feature    Impact Direction  Magnitude
# 0 Feature_A +0.35    🔴 Malware      0.35
# 1 Feature_B -0.18    🟢 Benign       0.18
```

### Create Waterfall
```python
fig = explainer.plot_waterfall(shap_exp)
fig.savefig('explanation.png')
```

---

## 📈 UI Layout

```
┌─────────────────────────────────────────────────────┐
│  🔍 Make Prediction Tab                              │
├─────────────────────────┬───────────────────────────┤
│                         │                            │
│  Enter Features         │  Prediction Result         │
│  ─────────────          │  ──────────────────        │
│  • Feature 1: 0.5       │  ✅ BENIGN                │
│  • Feature 2: 0.8       │  Confidence: 87.3%        │
│  • Feature 3: 0.3       │                            │
│  [🔮 Predict]           │  Malware Prob: 12.7%      │
│                         │  Benign Prob: 87.3%       │
│                         │                            │
│                         │  [📈 SHAP Explanation]    │
│                         │  ⏳ Computing...          │
│                         │                            │
│                         │  📊 Top Features:         │
│                         │  ─────────────────        │
│                         │  Feature | SHAP | Dir | Im │
│                         │  ─────────────────────────│
│                         │  (Table shown here)       │
│                         │                            │
│                         │  📈 Waterfall:            │
│                         │  (Plot shown here)        │
│                         │                            │
└─────────────────────────┴───────────────────────────┘
```

---

## ✅ Verification Checklist

- [ ] App running: `streamlit run app.py`
- [ ] Model trained in "📊 Train Model" tab
- [ ] Can make predictions in "🔍 Make Prediction" tab
- [ ] SHAP button clickable
- [ ] Features table displays (no errors)
- [ ] Waterfall plot renders (no errors)
- [ ] Red and blue bars visible
- [ ] Timing ~30-60 seconds
- [ ] Memory usage <500 MB
- [ ] All features labeled correctly

---

## 🎯 Key Concepts

### SHAP Value
- Quantifies feature's contribution to prediction
- Positive = Supports MALWARE classification
- Negative = Supports BENIGN classification
- Larger magnitude = More important

### KernelExplainer
- Model-agnostic (works with any model)
- Uses perturbation method
- Based on Shapley values (optimal)
- Trade-off: Accurate but slower

### Waterfall Plot
- Shows cumulative feature contributions
- Red = Malware indicators
- Blue = Benign indicators
- Horizontal bars = Easy to read

---

## 📞 Support

### Quick Links
- **Code**: `app.py` lines 474-527
- **SHAP Class**: `shap_explainer.py`
- **Tests**: `test_local_shap.py`
- **Docs**: `LOCAL_SHAP_IMPLEMENTATION_GUIDE.md`
- **Quick Start**: `LOCAL_SHAP_QUICK_START.md`

### Common Questions
Q: Why does SHAP take so long?  
A: KernelExplainer evaluates model on 50+ sample variations

Q: Can I make it faster?  
A: Use `explain_instance_fast()` for 15-30 second results

Q: What's a good SHAP value?  
A: >0.3 for malware, <-0.3 for benign (strong signals)

Q: Can I batch explain?  
A: Yes, CLI supports loops. UI explains one at a time

---

## 🚀 Ready to Use!

```
✅ LOCAL SHAP EXPLAINABILITY COMPLETE
✅ ALL TESTS PASSING (12/12)
✅ PRODUCTION READY
✅ FULLY DOCUMENTED

👉 Next: Open http://localhost:8504 and start exploring!
```

