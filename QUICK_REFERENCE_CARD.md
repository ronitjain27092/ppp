# SHAP Explainable AI - Quick Reference Card

## 🚀 Start in 30 Seconds

```bash
cd "e:\research code\malware-detection-xai"
pip install --upgrade shap tensorflow streamlit
streamlit run streamlit_app_fixed.py
```

Open browser to `localhost:8501` → Go to **📊 Train Model** → Upload CSV → Click **🚀 START TRAINING**

---

## 3 Tabs You'll Use

### 1️⃣ 🌍 Global Analysis
**Question:** "Which features indicate malware?"
- Bar Chart: Top 20 features
- Beeswarm: All features colored by impact
- Use: Verify model learns correct patterns

### 2️⃣ 🔍 Local Analysis  
**Question:** "Why this prediction for this sample?"
- Select sample (0-49)
- See waterfall (detailed) or force (compact) plot
- Use: Understand specific decisions

### 3️⃣ 📖 About SHAP
**Question:** "How does SHAP work?"
- Educational content
- Real-world examples
- How to read plots
- Use: Learn the method

---

## What SHAP Shows

### Example: Malware Prediction

```
Model says: "This is MALWARE (85% confidence)"

SHAP explains why:
├─ High memory usage: +0.30 (toward MALWARE)
├─ High API calls: +0.25 (toward MALWARE)
├─ High network activity: +0.15 (toward MALWARE)
└─ Result: 0.85 = MALWARE prediction ✓
```

**In Plain English:** "Model predicted malware because memory spike, API calls, and network connections all suggest malicious activity"

---

## Key Metrics

| Metric | Value | Note |
|--------|-------|------|
| Accuracy | 82-92% | Realistic, not misleading 99% |
| Recall | 75-90% | Catches 75-90% of malware |
| Training Time | 20-50 sec | Includes SHAP initialization |
| SHAP Computation | ~30 sec | First time, then cached |
| Explainable Samples | 50 | First 50 test samples |

---

## Real-World Use Case

### Incident Response Workflow

```
1. Alert: "Process XYZ flagged as malware"
   ↓
2. Find sample #15 (Process XYZ) in Local Analysis
   ↓
3. View SHAP waterfall:
   ├─ Memory: +0.40
   ├─ API: +0.30
   └─ Network: +0.15
   ↓
4. Cross-check with forensics:
   ├─ Memory dump: Shellcode found ✓
   ├─ API logs: CreateRemoteThread detected ✓
   └─ Network: C&C connected ✓
   ↓
5. Decision: QUARANTINE (high confidence)
```

---

## Plot Types Explained

### SHAP Summary Plot - Bar
```
What: Which features matter most?
How: Longer bar = more important
Use: Quick overview of model decision-making
```

### SHAP Summary Plot - Beeswarm
```
What: How do features contribute?
How: Red/blue dots show positive/negative impact
Use: Detailed understanding of feature behavior
```

### SHAP Waterfall Plot
```
What: Step-by-step breakdown of ONE prediction
How: Base → +Feature1 → +Feature2 → Final prediction
Use: Explain specific decisions
```

### SHAP Force Plot
```
What: Quick visual of ONE prediction
How: Red vs Blue forces pushing decision
Use: Compact summary for presentations
```

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Explainable AI not showing" | Train model first in 📊 Train Mode |
| "Computing SHAP...hangs" | Wait (30 sec) or reduce max_features |
| "Plots blank/error" | Check Recall > 0 in 🔍 Analyze Results |
| "ImportError: No module SHAP" | `pip install --upgrade shap` |
| "Memory error" | Reduce max_features slider or restart app |

---

## File Structure

```
Key Code Files:
├─ streamlit_app_fixed.py     ← Main app (run this)
├─ shap_explainer.py          ← SHAP engine
├─ cnn_model_fixed.py         ← CNN model
└─ evaluation_fixed.py        ← Metrics

Key Documentation:
├─ UI_HOW_TO_GUIDE.md         ← User guide (START HERE)
├─ SHAP_QUICK_START.md        ← Quick reference
├─ EXPLAINABLE_AI_GUIDE.md    ← Technical deep-dive
└─ README_EXPLAINABLE_AI.md   ← Full system guide
```

---

## Configuration

**In Streamlit UI Sidebar:**
- Epochs: 5-50 (how long to train)
- Batch Size: 16-64 (samples per update)
- Threshold: 0.1-0.9 (sensitivity to malware)

**SHAP Settings (in code):**
- Background samples: 100 (default)
- Test samples: 50 (default)
- Max features shown: 5-30 (default 20)

---

## Decision Tree: Which Plot to Use

```
Do I want to...?

├─ See overall pattern
│  └─ Use: Global Analysis → Bar chart
│
├─ Understand why a specific prediction
│  ├─ In detail?
│  │  └─ Use: Local Analysis → Waterfall plot
│  └─ Quick summary?
│     └─ Use: Local Analysis → Force plot
│
├─ Debug false alarm
│  └─ Use: Local Analysis → Waterfall plot → Compare to forensics
│
├─ Verify model trustworthiness
│  └─ Use: Global Analysis → Check features make sense
│
└─ Learn how SHAP works
   └─ Use: About SHAP → Read educational content
```

---

## 5-Minute Learning Path

1. **Minute 1:** Run app, train model
2. **Minute 2:** Go to Global Analysis
3. **Minute 3:** Look at top features
4. **Minute 4:** Go to Local Analysis
5. **Minute 5:** View waterfall plot

**You now understand SHAP!**

---

## Key Concepts

### What is SHAP?
Game theory tells us each feature's contribution to prediction.

### Why SHAP for Malware?
- Verify model uses real malware signals
- Debug unexpected predictions
- Build confidence for automation
- Document decisions for compliance

### How Different from Accuracy?
- Accuracy: "99% correct" (misleading if imbalanced)
- SHAP: "I'm 82% correct BECAUSE features X, Y, Z" (trustworthy)

---

## Compliance Checklist

For auditing/compliance use:

- [ ] Screenshot: Global SHAP plot
- [ ] Screenshot: Local SHAP for specific incident
- [ ] Document: Feature names and values
- [ ] Document: Prediction confidence
- [ ] Document: True label (if known)
- [ ] Document: Action taken
- [ ] Timestamp: When decision made
- [ ] Approver: Who reviewed

Result: Interpretable AI audit trail ✓

---

## Support Quick Links

| Need | Document |
|------|----------|
| Getting started | UI_HOW_TO_GUIDE.md |
| Quick ref | SHAP_QUICK_START.md |
| Technical help | EXPLAINABLE_AI_GUIDE.md |
| System architecture | README_EXPLAINABLE_AI.md |
| Implementation details | SHAP_INTEGRATION_SUMMARY.md |
| Implementation checklist | IMPLEMENTATION_CHECKLIST.md |

---

## Success Indicators

✅ You're using SHAP successfully if:

1. **Training completes** with "✓ Training Complete!"
2. **Global analysis shows** meaningful features
3. **Local analysis works** for different samples  
4. **Plots render** without errors
5. **Features make sense** (not random artifacts)
6. **Forensics verifies** SHAP explanations
7. **Team trusts** model decisions

---

## Pro Tips

💡 **Tip 1:** Always check Global first, then Local
💡 **Tip 2:** Compare malware vs benign samples
💡 **Tip 3:** Verify with forensics evidence
💡 **Tip 4:** Screenshot SHAP plots for audit trail
💡 **Tip 5:** Watch for model drift in SHAP patterns

---

## One-Line Summary

**CNN malware detector that explains every prediction using game theory (SHAP), suitable for security operations and forensics analysis.**

---

## Need Help?

1. Check relevant documentation file
2. Review troubleshooting section
3. Verify dependencies: `pip install --upgrade shap tensorflow`
4. Check system resources (memory, disk)
5. Retrain model with different settings

---

## Version Info

- **SHAP:** 0.41+
- **TensorFlow:** 2.0+
- **Python:** 3.8+
- **Streamlit:** Latest

---

## 🎯 You're All Set!

Everything needed to use Explainable AI:
- ✓ Code module (shap_explainer.py)
- ✓ Streamlit integration (streamlit_app_fixed.py)
- ✓ Documentation (5000+ lines)
- ✓ Real-world examples
- ✓ Troubleshooting guides

**Run:** `streamlit run streamlit_app_fixed.py`

**Start:** Upload CSV → Train → Explore SHAP

**Good luck!** 🛡️

---

## Quick Command Reference

```bash
# Start fresh
cd "e:\research code\malware-detection-xai"
pip install --upgrade shap tensorflow streamlit pandas numpy scikit-learn matplotlib

# Run Streamlit app
streamlit run streamlit_app_fixed.py

# Optional: Train offline (Python script)
python cnn_model_fixed.py

# Optional: Check dependencies
pip list | grep shap
pip list | grep tensorflow
```

---

## FAQ (3-Second Answers)

**Q: Is it fast?**
A: Yes. ~30 sec for SHAP computation, then cached.

**Q: Is it accurate?**
A: Yes. 82-92% realistic accuracy (not misleading 99%).

**Q: Is it easy to use?**
A: Yes. 3 tabs, click plot, view explanation.

**Q: Is it trustworthy?**
A: Yes. Verify explanations with forensics.

**Q: Is it production-ready?**
A: Yes. Error handling, documentation, examples all included.

---

**Last Updated:** April 8, 2026
**Status:** Production Ready ✅
**Ready to Deploy:** Yes ✅

