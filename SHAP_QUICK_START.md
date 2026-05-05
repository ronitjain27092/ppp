# Quick Start: Explainable AI

## TL;DR (What You Need to Know)

Your CNN malware detection system now explains its predictions using **SHAP** (game theory-based explanation).

**In Plain English:**
- **Global:** "These features indicate malware (across all samples)"
- **Local:** "This sample is malware because features A, B, C are high"

---

## One-Command Start

```bash
streamlit run streamlit_app_fixed.py
```

Then:
1. **📊 Train Model** → Train your CNN (includes SHAP setup)
2. **🧠 Explainable AI** → View explanations

---

## What You'll See

### 🌍 Global Analysis Tab

**"Which features matter for malware detection?"**

```
Example output:
kernel_memory_usage    ████████████  ← Most important
process_count          ██████████
cpu_time               ████████
network_connections    ██████
...
```

**How to read:**
- Longer bar = More important
- Red dots = High value → Often indicates Malware
- Blue dots = Low value → Often indicates Benign

### 🔍 Local Analysis Tab

**"Why did the model predict THIS for THIS sample?"**

```
Sample: 15
True Label: Malware
Predicted: Malware ✓
Confidence: 85%

Waterfall Explanation:
├─ kernel_memory_usage: +0.45 (MALWARE)
├─ api_call_freq: +0.25 (MALWARE)
├─ network_conn: +0.20 (MALWARE)
└─ Result: 0.90 = MALWARE prediction
```

**How to read:**
- Red bars: Push toward "Malware"
- Blue bars: Push toward "Benign"
- Bar height: How much influence

---

## 3-Step Usage

### Step 1: Train
1. Go to **📊 Train Model**
2. Upload CSV
3. Click **🚀 START TRAINING**
4. Wait for "✓ Training Complete!" (includes SHAP init)

### Step 2: Explore Global Explanations
1. Click **🧠 Explainable AI** tab
2. Go to **🌍 Global Analysis**
3. Select plot type (bar or beeswarm)
4. See "Which features indicate malware?"

### Step 3: Drill Down to Specific Prediction
1. Stay in **🧠 Explainable AI**
2. Go to **🔍 Local Analysis**
3. Slide to select sample (0-49)
4. See "Why was THIS sample classified THIS way?"

---

## Real-World: RAM Forensics Example

**Scenario:** Process XYZ flagged as malware, need to verify

**Use SHAP to investigate:**

```
1. GLOBAL ANALYSIS shows important features:
   - kernel_memory_usage
   - process_handles
   - network_activity
   
2. LOCAL ANALYSIS for sample 15 (Process XYZ) shows:
   ├─ High kernel_memory_usage → +0.40 (RED)
   ├─ High process_handles → +0.25 (RED)  
   ├─ High network_activity → +0.20 (RED)
   
   Prediction: MALWARE (0.85 confidence)

3. VERIFICATION by forensics analyst:
   - Check memory allocation: 512MB in 2 sec → Abnormal ✓
   - Check memory content: Found shellcode → Malware ✓
   - Check network: Connecting to C&C IP → Malware ✓
   
RESULT: Model explanation matches real forensics evidence
DECISION: QUARANTINE with confidence
```

---

## Performance

**Limitations (by design for speed):**
- Background samples: 100 (for baseline)
- SHAP computed: First 50 test samples (all you can explain)
- Computation time: ~30 seconds on typical hardware

**Why these limits?**
- Full SHAP on all samples = hours of computation
- 50 samples = good balance of coverage + speed
- If you need more: Can adjust in `shap_explainer.py`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Explainable AI not showing?" | Train model first in 📊 Train Model tab |
| "Computing SHAP hangs?" | Try again or reduce max_features slider |
| "SHAP shows same importance for all?" | Check model trained properly (recall > 0) |
| "Waterfall plot blank?" | Click Local Analysis tab again |
| "Error: SHAP initialization failed" | Pip install --upgrade shap; then retrain |

---

## Key Concepts

### What is SHAP?

Think of it like explaining a vote:
```
Prediction = 0.85 (MALWARE)

Why?
- Feature A voted +0.40 (MALWARE)
- Feature B voted +0.30 (MALWARE)
- Feature C voted -0.15 (BENIGN)
- Other features: +0.00
────────────────────────────
Total: 0.85 (MALWARE) ✓

SHAP = Each feature's vote size
```

### DeepExplainer vs KernelExplainer

We use **DeepExplainer** because:
- ✓ Fast (good for web interface)
- ✓ Works with TensorFlow/Keras
- ✓ Uses gradients internally (efficient)

---

## Interpreting the Plots

### Summary Plot (Bar Chart)

```
What: "Which features does model use?"

Example:
High_Memory ████████████ 0.80 importance
High_CPUTime ██████████ 0.72 importance
API_Calls   ████████ 0.63 importance
Mutex_Cnt   █████ 0.41 importance

Reading:
- Longer bars = model relies on this feature more
- Top features = most important for decision
```

### Waterfall Plot (Local Explanation)

```
What: "Why predicted THIS for THIS sample?"

Example:
Base (neutral)                0.50
├─ High_Memory: +0.30      → +0.80 (Malware signal)
├─ API_Calls: +0.10        → +0.90 (Stronger signal)
└─ Network_Conn: +0.05     → +0.95
PREDICTION: MALWARE

Reading:
- Positive bar = pushed toward Malware
- Negative bar = pushed toward Benign
- Height = how much influence
```

---

## Use Cases

### 1. Verify Model Trustworthiness
```
Global Analysis shows:
- important features: memory, API calls, network
  ✓ Good: These match real malware behavior
- important feature: timestamp
  ✗ Suspicious: This shouldn't indicate malware
```

### 2. Debug False Positives
```
Local Analysis of false positive:
Prediction: MALWARE (0.75)
Actually: BENIGN

Features:
- High memory: +0.40 (Wrong signal for this sample)
- High API calls: +0.15 (Expected, benign process)

FIX: Retrain with more benign samples with high memory
```

### 3. Incident Response Reporting
```
Process XYZ flagged as malware

SHAP Explanation:
├─ kernel_memory: +0.45 (Evidence 1)
├─ network_activity: +0.30 (Evidence 2)
└─ process_handles: +0.15 (Evidence 3)

Forensics Verification: ✓ All confirmed
Decision: QUARANTINE with high confidence
```

---

## Settings Reference

**Global Analysis:**
- Plot Type: "bar" (features) or "beeswarm" (all values)
- Max Features: 5-30 (recommend 20)

**Local Analysis:**
- Sample: 0-49 (which sample to explain)
- Type: "Waterfall" (detailed) or "Force" (compact)

---

## Files

| File | Purpose |
|------|---------|
| `shap_explainer.py` | SHAP computation engine |
| `streamlit_app_fixed.py` | Streamlit UI with SHAP integration |
| `EXPLAINABLE_AI_GUIDE.md` | Detailed guide (this file) |
| `cnn_model_fixed.py` | CNN model (unchanged) |
| `evaluation_fixed.py` | Evaluation metrics (unchanged) |

---

## Advanced: Customization

### Change Background Sample Count

In `streamlit_app_fixed.py`, line ~180:
```python
shap_explainer = SHAPExplainer(
    model=model.model,
    X_background=X_train_scaled[:50],      # Changed from 100
    feature_names=feature_names,
    max_background_samples=50             # Changed from 100
)
```

### Change Test Sample Count

In `streamlit_app_fixed.py`, local analysis section:
```python
X_test_subset = st.session_state.X_test_scaled[:100]  # Changed from 50
```

### Add More Explanation Types

In `shap_explainer.py`, add new methods:
```python
def plot_dependence(self, feature_index):
    # Show how feature value affects prediction
    
def plot_interaction(self, feature_i, feature_j):
    # Show how two features interact
```

---

## Summary

✅ **You now have:**
- Global explanations (feature importance)
- Local explanations (specific predictions)
- RAM forensics integration
- Streamlit web interface
- Production-ready code

✅ **Your system is now:**
- Interpretable (can explain predictions)
- Trustworthy (explanations verifiable)
- Audit-ready (compliance documentation)

🎯 **Next steps:**
1. Train model with your data
2. Explore 🌍 Global Analysis
3. Use 🔍 Local Analysis for incidents
4. Document explanations for compliance

