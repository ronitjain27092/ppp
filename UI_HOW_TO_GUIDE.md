# How to Use Explainable AI - UI Guide

## Start Here: 3 Simple Steps

### Step 1️⃣: Train the Model

```
1. Open Streamlit app (localhost:8501)
2. Click "📊 Train Model" in sidebar
3. Upload your CIC-MalMem CSV file(s)
4. Click "🚀 START TRAINING"
5. Wait for "✓ Training Complete!" message
   (Progress bar shows: Preprocessing → Build → Class Weights → Train → Evaluate → SHAP Init → Complete)
```

**What's happening:**
- Data is preprocessed (split, scaled)
- Neural network is trained
- Confidence is calculated
- SHAP explainer is initialized (ready to explain)

**Status indicators:**
- ✅ Green: Feature working
- ⚠️ Yellow: Warning (may be recoverable)
- ❌ Red: Error (check troubleshooting)

---

### Step 2️⃣: View Explanations

Once training completes:

```
1. Click "🧠 Explainable AI" in sidebar
2. Three tabs appear:
   - 🌍 Global Analysis (which features matter?)
   - 🔍 Local Analysis (why this prediction?)
   - 📖 About SHAP (learn about the method)
```

---

### Step 3️⃣: Explore & Understand

#### Tab 1: Global Analysis 🌍

**Question:** "Which features does the model use to detect malware?"

**How to use:**

```
1. Select plot type:
   - "bar (Top Features)" 
     → Shows most important features
     → Good for: Quick overview
   
   - "beeswarm (Feature Impact)"
     → Shows all feature values colored by importance
     → Good for: Detailed analysis

2. Move "Max Features" slider (default: 20)
   - More features: More detailed but harder to read
   - Fewer features: Simpler but less information

3. Click plot to regenerate (~30 seconds)

4. Read the plot:
   - Longer bars = More important features
   - Red dots = High values (often Malware)
   - Blue dots = Low values (often Benign)
```

**Example Output:**
```
kernel_memory_usage    ████████████  ← Most important
process_count          ██████████
cpu_time               ████████
network_conn           ██████

Interpretation:
"Model uses memory behavior and process generation
to detect malware. This matches real malware behavior!"
```

---

#### Tab 2: Local Analysis 🔍

**Question:** "Why did the model predict THIS for THIS specific sample?"

**How to use:**

```
1. Left side - Configuration:
   
   a) "Select Sample:" Slider
      - Move to pick sample 0-49
      - Each sample = one analyzed process
   
   b) "Explanation Type:" Radio button
      - "Waterfall (Detailed)"
        → Step-by-step breakdown
        → Shows all feature contributions
      - "Force (Compact)"
        → Quick summary visualization
        → Red/blue forces pushing decision

2. Middle - Sample Information:
   Shows:
   - True Label: What was it really? (Benign or Malware)
   - Prediction: What did model say? (Benign or Malware)
   - Confidence: How sure (0-100%)
   - Malware Probability: 0-1 (model's output)

3. Right side - Explanation:
   - Waterfall plot OR
   - Force plot
   (Choose which in configuration)
```

**Example Output:**

```
Sample Info:
- True Label: Malware
- Prediction: Malware ✓ CORRECT
- Confidence: 85%
- Malware Probability: 0.85

Waterfall Plot:
Base prediction: 0.50 (neutral)
├─ kernel_memory: +0.30 (RED → Malware direction)
├─ process_handles: +0.15 (RED → Malware)
├─ api_calls: +0.10 (RED → Malware)
└─ Final prediction: 0.85 (MALWARE)

Interpretation:
"Model predicted Malware because:
1. HIGH memory usage (abnormal)
2. HIGH process handles (suspicious)
3. HIGH API activity (typical malware)

All indicators point to malware
Evidence is strong & aligned"
```

---

#### Tab 3: About SHAP 📖

**Question:** "What is SHAP and why should I care?"

**Content includes:**
- Simple explanation of how SHAP works
- Why it's important for malware detection
- How to use it in forensics analysis
- How to read the different plot types
- Compliance benefits
- Real-world examples

**Read this if:**
- You want to understand the method
- You need to explain it to others
- You want background knowledge
- You're evaluating model trustworthiness

---

## Real-World Usage Examples

### Example 1: Verify Model is Trustworthy

```
Scenario: Security team wants to trust the model

Step 1: Go to Global Analysis tab
Step 2: View SHAP summary plot
Step 3: Check features:
  ✓ Good: memory_usage, api_calls, network_conn
          (these ARE indicators of malware)
  ✗ Bad: timestamp, color_of_window
         (these should NOT be important)

Decision:
✓ Model learns real patterns
✓ Trustworthy for use in operations
```

### Example 2: Incident Response

```
Scenario: Process flagged as malware, need to verify

Step 1: Find sample in Global Analysis
Step 2: Go to Local Analysis tab
Step 3: Select sample (e.g., sample #12)
Step 4: Review SHAP waterfall showing:
  - memory: +0.40 (abnormal)
  - api_calls: +0.30 (malicious)
  - network: +0.15 (suspicious)

Step 5: Verify with forensics
  - Check memory dump: Contains shellcode? YES ✓
  - Check API logs: CreateRemoteThread? YES ✓
  - Check network: C&C server? YES ✓

Conclusion:
Evidence from SHAP matches forensics analysis
Decision: QUARANTINE (high confidence)
```

### Example 3: Debug False Positive

```
Scenario: Model predicted MALWARE but it's BENIGN

Step 1: Go to Local Analysis
Step 2: Select the false positive sample
Step 3: Review SHAP waterfall:
  - memory: +0.35 (HIGH)
  - api_calls: -0.20 (LOW)
  - network: -0.10 (LOW)

Analysis:
Memory is high, but this is legitimate process
This process SHOULD have high memory usage

Investigation:
✓ Look at data: Are we mislabeling benign as malware?
✓ Check preprocessing: Is memory scaled correctly?
✓ Review training: Do we have enough benign samples with high memory?

Action:
Retrain with better data quality
```

---

## Understanding the Plots

### SHAP Summary Plot - Bar Chart

```
What you see:
kernel_memory_usage    ████████████
process_count          ██████████
cpu_time               ████████

What it means:
- Longer bar = Model relies on this feature more
- Top features = Most important decisions
- Bottom features = Least important

How to read it:
"Memory usage is the top signal for malware detection"

Use case:
"Is the model using the right features?"
```

---

### SHAP Summary Plot - Beeswarm Chart

```
What you see:
SHAP value ranges (left=negative, right=positive)
Each dot = one sample's SHAP contribution
Color shows: Red (high value) to Blue (low value)

What it means:
- Right side (red): High value pushed toward MALWARE
- Left side (blue): Low value pushed toward BENIGN
- Spread: How consistent is this feature?

How to read it:
"When memory usage is high (red), it pushes toward malware"

Use case:
"How consistent is this feature's behavior?"
```

---

### SHAP Waterfall Plot - Local Explanation

```
What you see:
Base value: 0.50
├─ Feature1: +0.30
├─ Feature2: +0.10
└─ Final: 0.90

What it means:
- Base (0.50): Neutral starting point
- Red bars (+): Push toward MALWARE
- Blue bars (-): Push toward BENIGN
- Height: Strength of push
- Flow left-to-right: Step-by-step buildup

How to read it:
"This sample predicted MALWARE because..."
"Feature1 contributed +0.30"
"Feature2 contributed +0.10"
"Combined = 0.90 = MALWARE"

Use case:
"Why was THIS sample classified THIS way?"
```

---

### SHAP Force Plot - Local Explanation (Compact)

```
What you see:
← Blue (Benign) | Red (Malware) →

What it means:
- Red force: Features pushing toward MALWARE
- Blue force: Features pushing toward BENIGN
- Width: Strength of each push
- Direction: Which way wins

How to read it:
"Red forces are stronger, so prediction = MALWARE"

Use case:
"Quick summary of why this prediction"
```

---

## Troubleshooting

### Problem: "Explainable AI tab not showing"

**Cause:** Model not trained yet

**Solution:**
```
1. Go to "📊 Train Model"
2. Upload CSV file
3. Click "🚀 START TRAINING"
4. Wait for "✓ Training Complete!"
5. Then return to "🧠 Explainable AI"
```

---

### Problem: "Computing SHAP...spinner keeps spinning"

**Cause:** First-time SHAP computation is slow

**What's happening:**
```
- Computing SHAP values for 50 samples
- Using 100 background samples
- Takes ~30 seconds on typical hardware
```

**What to do:**
```
Option 1: Wait (it will finish)
Option 2: Reduce max_features slider to 10
Option 3: Restart and check system resources
```

---

### Problem: "Plot is blank or shows error"

**Cause:** Model predictions all one class (only Benign or only Malware)

**Check:**
```
1. Go to "🔍 Analyze Results"
2. Look at metrics:
   - Is Recall > 0? (No = only predicting Benign)
   - Is F1 > 0? (No = problem)
3. Check Confusion Matrix:
   - Should show all 4 boxes (TP, FP, FN, TN)
   - If only top-left: Only predicting Benign ✗
```

**Solution:**
```
1. Go back to "📊 Train Model"
2. Try different settings:
   - Increase epochs (20 → 30)
   - Try different batch size (32 → 16 or 64)
   - Lower threshold slider (0.3 → 0.2)
3. Retrain
4. Check metrics again
```

---

### Problem: "Memory error" or "System unresponsive"

**Cause:** Too many samples being processed

**Solution:**
```
In Local Analysis tab:
1. Reduce max_features slider (20 → 10)
2. Try different sample (smaller index)

Or restart app:
1. Press Ctrl+C in terminal to stop Streamlit
2. Run: streamlit run streamlit_app_fixed.py
3. Retrain with smaller dataset
```

---

## Tips & Tricks

### Tip 1: Use Global Analysis First
```
Before looking at specific predictions:
1. Go to Global Analysis
2. Understand what features matter
3. Check if they make sense
4. Then explore specific samples
```

### Tip 2: Compare Malware vs Benign
```
1. Global Analysis: See general patterns
2. Use Local Analysis to compare:
   - Sample 5: TRUE MALWARE → Why malware?
   - Sample 10: TRUE BENIGN → Why benign?
3. Understand differences
```

### Tip 3: Investigate False Positives
```
When model predicts wrong:
1. Local Analysis on the false positive
2. See what features led to wrong prediction
3. Question: Do those features really indicate malware?
   - If NO: Model may be learning wrong patterns
   - If YES: Is sample mislabeled?
4. Debug accordingly
```

### Tip 4: Document for Compliance
```
When making security decision:
1. Take screenshot of SHAP plot
2. Note the confidence percentage
3. Add to incident report
4. Provides audit trail
```

---

## FAQ

### Q: Which plot should I use - Waterfall or Force?

**A:** 
```
Use WATERFALL:
- Need detailed step-by-step explanation
- Want to see all features
- For documentation/reporting
- For forensics analysis

Use FORCE:
- Need quick summary
- For presentations
- For quick decisions
- Space limited
```

### Q: How many features should I display?

**A:**
```
- 5-10: Too simple, missing details
- 15-20: Good balance (recommended)
- 25-30: Detailed but cluttered
- >30: Too much info, hard to understand

Recommendation: Start with 20, adjust as needed
```

### Q: Why are some samples not explainable?

**A:**
```
SHAP can only explain samples 0-49 (first 50 test samples)
This is by design for speed
- Computing SHAP for all 1000+ test samples = hours
- First 50 = good representative sample = 30 seconds
```

### Q: Can I change the threshold after training?

**A:**
```
In sidebar:
- "Prediction Threshold" slider
- Default: 0.3
- Range: 0.1-0.9

Changing threshold changes:
- Which samples predicted as malware/benign
- BUT: Doesn't recompute SHAP (stays on same samples)

Use case:
"What if I used threshold 0.2 instead?"
View different predictions without retraining
```

### Q: How do I know if SHAP explanations are correct?

**A:**
```
Verify with domain knowledge:
1. Global Analysis shows important features
   → Do they match known malware behavior?
   
2. Local Analysis for specific sample
   → Do the positive features actually present in sample?
   → Can forensics analyst confirm?

Example:
SHAP says: "High memory + high API = Malware"
Forensics confirms: Memory dump has shellcode, API calls detected
Result: Explanation is correct ✓
```

---

## Next Steps After Using Explainable AI

### For Security Teams

1. **Initial Setup:**
   - Train model with your data
   - Review Global Analysis
   - Verify model trustworthiness

2. **Operational Use:**
   - Use Local Analysis for alerts
   - Document with SHAP plots
   - Build confidence base

3. **Integration:**
   - Export SHAP explanations
   - Add to incident tickets
   - Build audit trail

### For Forensics Analysts

1. **Understand Patterns:**
   - Review Global Analysis
   - See which memory features indicate malware

2. **Cross-Reference:**
   - For each prediction
   - Get SHAP explanation
   - Verify with memory dump analysis

3. **Documentation:**
   - Screenshot SHAP plots
   - Add to forensics report
   - Provides evidence

### For Compliance/Audit

1. **Collect Evidence:**
   - SHAP global summary
   - SHAP local for important cases
   - Screenshots of decisions

2. **Audit Trail:**
   - Document when predictions used
   - Show explanation reasons
   - Timestamp everything

3. **Compliance Report:**
   - System is interpretable (SHAP) ✓
   - Decisions are documented ✓
   - Evidence is provided ✓
   - Ready for review ✓

---

## Summary

**You now know how to:**

✅ Train the CNN model with your data
✅ View global explanations (features importance)
✅ View local explanations (specific predictions)
✅ Understand SHAP plots
✅ Use explanations in operations
✅ Debug model behavior
✅ Document for compliance

**Questions?** Check the detailed guides:
- EXPLAINABLE_AI_GUIDE.md (comprehensive)
- SHAP_QUICK_START.md (quick reference)
- README_EXPLAINABLE_AI.md (full system)

**Ready to use it?** Run:
```bash
streamlit run streamlit_app_fixed.py
```

Good luck! 🛡️

