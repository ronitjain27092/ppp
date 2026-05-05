# Running Malware Detection XAI in VS Code - Complete Guide

## 📍 Prerequisites

- VS Code installed ([Download](https://code.visualstudio.com/))
- Python 3.8+ installed ([Download](https://www.python.org/))
  - ✅ Check "Add Python to PATH" during installation
- CIC-MalMem-2022 CSV dataset (optional for testing)

## 🎯 Part 1: Open Project in VS Code

### Method 1: VS Code File Menu (Easiest)
1. Open VS Code
2. **File** → **Open Folder**
3. Navigate to: `e:\research code\malware-detection-xai`
4. Click **Select Folder**
5. Wait for VS Code to load the project

### Method 2: Command Line
```powershell
cd "e:\research code\malware-detection-xai"
code .
```

### Verify Project Structure
In VS Code, you should see:
```
📦 malware-detection-xai
 ├── 📄 app.py
 ├── 📄 preprocessing.py
 ├── 📄 model.py
 ├── 📄 explain.py
 ├── 📄 requirements.txt
 ├── 📄 README.md
 ├── 📄 QUICK_START.md
 ├── 📄 setup.bat
 └── 📄 test_installation.py
```

---

## 🐍 Part 2: Configure Python Environment in VS Code

### Step 1: Open Terminal in VS Code
Press: **Ctrl + `** (backtick key)

You'll see PowerShell terminal at bottom of VS Code.

### Step 2: Create Virtual Environment
Copy-paste this command:

```powershell
python -m venv venv
```

Wait 30 seconds. You'll see a new `venv` folder in the project.

### Step 3: Activate Virtual Environment
Copy-paste:

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` prefix in terminal:
```
(venv) PS e:\research code\malware-detection-xai>
```

### Step 4: Select Python Interpreter in VS Code
1. Press: **Ctrl + Shift + P**
2. Type: `Python: Select Interpreter`
3. Choose: "./venv/bin/python" (this is the virtual environment Python)
4. Verify terminal shows `(venv)` prefix

### Step 5: Install All Dependencies
In the terminal, copy-paste:

```powershell
pip install -r requirements.txt
```

**⏳ This takes 5-10 minutes**

Watch terminal for:
- Downloading packages
- Installing TensorFlow (largest, ~500MB)
- Final message shows successful installations

### Verify Installation
Test if everything installed:

```powershell
python -c "import tensorflow; import shap; import streamlit; print('✓ All installed!')"
```

You should see: `✓ All installed!`

---

## ✅ Part 3: Verify with Test Script

Before running the app, test the installation:

```powershell
python test_installation.py
```

This will:
- Check Python version ✓
- Test imports ✓
- Create sample data ✓
- Train mini model ✓
- Test SHAP ✓

You'll see: `✓ ALL TESTS PASSED!`

If any test fails, see **Troubleshooting** section below.

---

## 🚀 Part 4: Run the Streamlit Application

With virtual environment activated, run:

```powershell
streamlit run app.py
```

You'll see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Your browser automatically opens the app!** 🎉

### Using the App

#### 👉 First Tab: Train Model
1. Click "Browse files"
2. Choose your CIC-MalMem-2022 CSV file
3. Adjust settings on right (optional)
4. Click blue "🚀 Start Training" button
5. Watch progress in console
6. See results with visualizations

#### 👉 Second Tab: Make Predictions
1. Enter feature values or upload CSV
2. Click "🔮 Predict"
3. See SHAP explanation
4. View contributing features

#### 👉 Third Tab: Model Analysis
- See top important features
- Analyze global model behavior
- Review metrics

---

## 🗂️ Part 5: Using Individual Python Files

### Edit and Run Individual Scripts

#### A. Just Preprocess Data

Create file `test_preprocessing.py`:

```python
from preprocessing import DataPreprocessor

# Load and preprocess your data
preprocessor = DataPreprocessor()
X, y, features, df = preprocessor.preprocess("path/to/your/dataset.csv")

print(f"Features shape: {X.shape}")
print(f"Feature names: {features}")
```

Run it:
```powershell
python test_preprocessing.py
```

#### B. Train Model Only

Create file `train_model.py`:

```python
from preprocessing import DataPreprocessor
from model import MalwareDetectionModel

# Preprocess
print("Preprocessing...")
preprocessor = DataPreprocessor()
X, y, features, df = preprocessor.preprocess("dataset.csv")

# Train model
print("Training model...")
model = MalwareDetectionModel(model_type='dnn')
X_train, X_test, y_train, y_test = model.split_data(X, y)
model.create_model(X_train.shape[1])
model.train(epochs=50, batch_size=32)

# Save
model.save_model("trained_model.h5")
print("✓ Model saved!")
```

Run it:
```powershell
python train_model.py
```

#### C. Generate Explanations

Create file `explain_model.py`:

```python
from model import MalwareDetectionModel
from explain import SHAPExplainer
from preprocessing import DataPreprocessor

# Load and preprocess
preprocessor = DataPreprocessor()
X, y, features, df = preprocessor.preprocess("dataset.csv")

# Load trained model
model = MalwareDetectionModel()
model.load_model("trained_model.h5")
model.X_test = X  # Set test data

# Create SHAP explainer
explainer = SHAPExplainer(model.model, features)
explainer.create_explainer(X)

# Get importance
importance = explainer.get_feature_importance(X)
print(importance.head(10))
```

---

## 🐛 Troubleshooting in VS Code

### Issue 1: "Python not found"

**In VS Code Terminal:**
```powershell
python --version
```

If this fails:
1. **File** → **Preferences** → **Settings**
2. Search: `python.defaultInterpreterPath`
3. Set to: `C:\Users\YourUsername\AppData\Local\Programs\Python\Python311\python.exe`

(Replace `YourUsername` and version as needed)

### Issue 2: "Module not found" errors

Make sure virtual environment is activated:

✅ Good (you see prefix):
```
(venv) PS e:\research code\malware-detection-xai> 
```

❌ Bad (no prefix):
```
PS e:\research code\malware-detection-xai>
```

Activate it:
```powershell
.\venv\Scripts\Activate.ps1
```

### Issue 3: "Permission denied" when activating

Try this:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Issue 4: Streamlit won't open browser

Manually open:
1. See terminal message: `Local URL: http://localhost:8501`
2. Open browser
3. Go to: `http://localhost:8501`

Or use:
```powershell
streamlit run app.py --logger.level=debug
```

### Issue 5: "TensorFlow not found" after pip install

TensorFlow is large, reinstall:
```powershell
pip uninstall tensorflow -y
pip install tensorflow==2.13.0
```

### Issue 6: Out of memory during training

1. In app: reduce "Epochs" (try 20)
2. Reduce "Test Size" (try 0.1)
3. In model.py: change line ~94:
   ```python
   hidden_layers=[64, 32, 16]  # Smaller layers
   ```

### Issue 7: Dataset not uploading

Check:
1. File is `.csv` format (not `.xlsx` or `.xls`)
2. File has label column: "Class", "Label", "Target", etc.
3. File is not too large (>500MB might have issues)
4. File has no special characters in column names

---

## 🎯 Quick Reference - Common Commands

### In VS Code Terminal (with `(venv)` prefix)

```powershell
# Run the app
streamlit run app.py

# Test installation
python test_installation.py

# Run custom script
python your_script.py

# Check installed packages
pip list

# Update package
pip install --upgrade tensorflow

# Exit virtual environment
deactivate

# Reactivate later
.\venv\Scripts\Activate.ps1
```

---

## 📊 Part 6: Workflow Example

### Complete Example from Start to Finish

#### Step 1: Open VS Code
```
VS Code → Open Folder → select malware-detection-xai → ✓
```

#### Step 2: Setup (one time only)
In VS Code terminal:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Step 3: Verify installation
```powershell
python test_installation.py
# You should see: ✓ ALL TESTS PASSED!
```

#### Step 4: Run the app
```powershell
streamlit run app.py
# Browser opens automatically
```

#### Step 5: Use the app
- Upload your CIC-MalMem-2022 CSV
- Set model type to "dnn"
- Set epochs to 50
- Click "🚀 Start Training"
- Wait for completion
- View metrics and plots

#### Step 6: Make predictions
- Go to "🔍 Make Prediction" tab
- Enter feature values or upload CSV
- Click "🔮 Predict"
- See SHAP explanation

---

## 🔍 Debugging Tips

### See What Python is Running
```powershell
which python  # or
Get-Command python
```

### See Installed Packages
```powershell
pip list | grep -E "tensorflow|shap|streamlit"
```

### Test Individual Imports
```powershell
python -c "import shap; print(shap.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Run Script with Full Debug Output
```powershell
streamlit run app.py --logger.level=debug
```

### Check System Resources
```powershell
Get-Process python  # See running Python processes
```

---

## ✨ Pro Tips for VS Code

### 1. Format Code (Optional)
Install extension: **Python** (Microsoft)
Then: **Right-click** → **Format Document**

### 2. Debugging
Add file `debug.py`:
```python
from preprocessing import DataPreprocessor
# Your test code here
```

Then press **F5** to debug.

### 3. Multiple Terminals
- Click ➕ in terminal tab
- Open separate terminal for monitoring

### 4. View Output
Run script, then check:
- **View** → **Output** → **Terminal**

### 5. Keyboard Shortcuts
- **Ctrl + Shift + P**: Command palette
- **Ctrl + `**: Toggle terminal
- **F5**: Debug
- **Ctrl + K Ctrl + 0**: Collapse all folders

---

## 📚 Additional Resources

- **VS Code Python**: https://code.visualstudio.com/docs/languages/python
- **Streamlit Docs**: https://docs.streamlit.io/
- **TensorFlow Guide**: https://tensorflow.org/guide
- **SHAP Library**: https://github.com/slundberg/shap

---

## ✅ Final Checklist

- [ ] VS Code installed and opened
- [ ] Malware-detection-xai folder opened
- [ ] Virtual environment created and activated (`(venv)` visible)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Test script passed (`python test_installation.py`)
- [ ] Streamlit app runs (`streamlit run app.py`)
- [ ] Browser opens to localhost:8501
- [ ] CSV dataset ready to upload
- [ ] All tabs visible (Train, Predict, Analyze)

**All checked? You're ready!** 🚀

---

**Support**: Check README.md for detailed documentation
**Status**: Ready to detect malware with explainable AI! 🛡️
