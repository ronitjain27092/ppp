# QUICK START GUIDE - 5 Minutes to Running

## ⚡ The Fastest Way to Get Started

### Step 1: Extract & Navigate (30 seconds)
```powershell
cd "e:\research code\malware-detection-xai"
```

### Step 2: Run Setup Script (5 minutes)
Double-click: **`setup.bat`**

This automatically:
- Creates Python virtual environment
- Installs all dependencies
- Verifies everything works

**Wait for completion!** You'll see "SUCCESS!" message.

### Step 3: Launch Application (10 seconds)
```powershell
# Copy-paste this into PowerShell:

.\venv\Scripts\Activate.ps1
streamlit run app.py
```

**Done!** Browser opens automatically to the app.

---

## 🎯 What to Do Next

### First Time Users

1. **Have your CIC-MalMem-2022 CSV ready**
   - File should have features + "Class" column
   - Benign/Malware labels

2. **Go to "📊 Train Model" tab**
3. **Click "Browse files" → Select your CSV**
4. **Click "🚀 Start Training"**
5. **Wait for training to complete** (2-5 minutes)
6. **See results and metrics!**

### Next: Make Predictions

1. **Go to "🔍 Make Prediction" tab**
2. **Choose "Manual Input" or "Upload File"**
3. **Enter values or upload CSV**
4. **Click "🔮 Predict"**
5. **See SHAP explanation of prediction!**

### Advanced: Analyze Model

1. **Go to "📈 Model Analysis" tab**
2. **View feature importance**
3. **Understand which features matter most**

---

## 🆘 If Something Goes Wrong

### Problem: Python not found
```
Solution: Install Python from python.org
Make sure "Add to PATH" is checked during install
```

### Problem: setup.bat didn't work
Manual installation:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Problem: Streamlit won't start
```powershell
# Try this:
streamlit run app.py --logger.level=debug

# Or check Python is correct version:
python --version  # Should be 3.8+
```

### Problem: "Module not found"
```powershell
# Reinstall packages:
pip install -r requirements.txt --upgrade
```

---

## 📊 Dataset Format

Your CSV must look like this:

| Feature1 | Feature2 | Feature3 | ... | Class    |
|----------|----------|----------|-----|----------|
| 0.5      | 0.3      | 0.8      | ... | Benign   |
| 0.6      | 0.4      | 0.9      | ... | Malware  |

✅ Label names that work:
- `Class` (best)
- `Label`
- `Target`
- `Type`
- `Category`

✅ Label values:
- Benign: "Benign", "Normal", "0"
- Malware: "Malware", "Malicious", "1"

---

## 🎓 Understanding Results

After training you'll see:

- **Accuracy**: How many predictions are correct (0-1)
- **Precision**: Of detected malware, how many are real (0-1)
- **Recall**: Of actual malware, how many we found (0-1)
- **F1-Score**: Balance between precision & recall (0-1)
- **ROC-AUC**: Overall model quality (0-1)

**Good Results**: All metrics > 0.90

---

## 🚀 Advanced: Command Line Use

### Train model from command line

```python
# Create train.py:
from preprocessing import DataPreprocessor
from model import MalwareDetectionModel

preprocessor = DataPreprocessor()
X, y, features, df = preprocessor.preprocess("dataset.csv")

model = MalwareDetectionModel(model_type='dnn')
X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2)
model.create_model(X_train.shape[1])
model.train(epochs=50)

model.save_model("my_model.h5")
```

Run with:
```powershell
python train.py
```

---

## 📁 File Structure

```
malware-detection-xai/
├── setup.bat              ← Run this first!
├── requirements.txt       ← Dependencies
├── app.py                 ← Web interface
├── preprocessing.py       ← Data handling
├── model.py               ← Deep learning
├── explain.py             ← Explainability
└── README.md              ← Full documentation
```

---

## 💡 Pro Tips

1. **First run slower?** TensorFlow needs setup time (normal)
2. **GPU available?** Good! TensorFlow will auto-detect
3. **Large dataset?** Start with small subset for testing
4. **Want different model?** Change "Model Type" to "cnn"
5. **Predictions missing?** Check feature column names match

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] setup.bat run successfully
- [ ] Streamlit app opens in browser
- [ ] CSV dataset ready
- [ ] Training complete
- [ ] Predictions working
- [ ] SHAP explanations visible

**All set?** You're ready for cybersecurity AI! 🛡️

---

Need help? Check README.md for detailed documentation.
