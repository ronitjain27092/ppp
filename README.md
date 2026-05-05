# Explainable AI-Based RAM Forensics for Malware Detection

A complete deep learning project for detecting malware in RAM forensics data using explainable AI (SHAP) for interpretability.

## 📋 Project Structure

```
malware-detection-xai/
├── preprocessing.py          # Data preprocessing and normalization
├── model.py                  # Deep Learning models (DNN & CNN)
├── explain.py                # SHAP explainability module
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎯 Features

- **Automatic Dataset Detection**: Automatically detects label columns (Class, Label, Type, etc.)
- **Data Preprocessing**: 
  - Handles missing values intelligently
  - Normalizes features using MinMaxScaler
  - Binary label conversion (Benign=0, Malware=1)

- **Deep Learning Models**:
  - Deep Neural Network (DNN) with batch normalization and dropout
  - Convolutional Neural Network (CNN) for sequential feature processing
  - Early stopping and validation monitoring

- **Explainable AI**:
  - SHAP (SHapley Additive exPlanations) for model interpretability
  - Global feature importance analysis
  - Per-instance explanations

- **Visualizations**:
  - Confusion matrix
  - ROC curve with AUC score
  - Training history (accuracy & loss)
  - SHAP summary plots
  - Feature dependence plots

- **Web Interface**:
  - Streamlit-based interactive UI
  - Dataset upload functionality
  - Real-time predictions
  - Interactive visualization dashboard

## 📦 Requirements

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`

## 🚀 Quick Start Guide

### Step 1: Set Up Python Environment

**Option A: Using venv (Recommended)**

```powershell
# Navigate to project directory
cd "e:\research code\malware-detection-xai"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Windows CMD:
venv\Scripts\activate.bat

# On Mac/Linux:
source venv/bin/activate
```

**Option B: Using conda**

```powershell
conda create -n malware-detection python=3.9
conda activate malware-detection
```

### Step 2: Install Dependencies

```powershell
# Ensure you're in the project directory with activated virtual environment
pip install -r requirements.txt
```

This will install:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities
- `tensorflow` - Deep learning framework
- `shap` - Explainability library
- `matplotlib` & `seaborn` - Visualization
- `streamlit` - Web interface

**Note**: First installation may take 5-10 minutes. TensorFlow is the largest dependency (~500MB).

### Step 3: Run the Application

```powershell
# Make sure virtual environment is activated
streamlit run app.py
```

This will:
- Start the Streamlit server
- Open the web interface in your default browser (usually `http://localhost:8501`)
- Display a beautiful interactive dashboard

## 📊 Using the Application

### Mode 1: Train Model ('📊 Train Model' tab)

1. **Upload Dataset**:
   - Click "Browse files" to upload your CIC-MalMem-2022 CSV file
   - File must contain features and a label column

2. **Configure Settings** (right sidebar):
   - **Model Type**: Choose between DNN (recommended) or CNN
   - **Test Size**: Percentage of data to use for testing (default: 20%)
   - **Epochs**: Number of training iterations (default: 50)

3. **Start Training**:
   - Click "🚀 Start Training" button
   - Monitor progress in the console
   - Wait for training to complete

4. **View Results**:
   - Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics
   - Training history plot (accuracy & loss curves)
   - Confusion matrix heatmap
   - ROC curve visualization

### Mode 2: Make Prediction ('🔍 Make Prediction' tab)

**Method 1: Manual Input**
- Enter values (0.0 to 1.0) for each feature
- Click "🔮 Predict" to get prediction
- View SHAP explanation and contributing features

**Method 2: Batch Prediction**
- Upload CSV file with multiple samples
- Click "🔮 Predict All Samples"
- Get predictions for all samples with confidence scores

### Mode 3: Model Analysis ('📈 Model Analysis' tab)

- View top features by importance
- See feature importance rankings
- Analyze global model behavior through SHAP summaries

## 🔧 Running Individual Modules (Programmatic Usage)

### Preprocessing Only

```python
from preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
X, y, feature_names, df = preprocessor.preprocess("path/to/dataset.csv")

print(f"Features shape: {X.shape}")
print(f"Labels unique values: {y.unique()}")
print(f"Feature names: {feature_names}")
```

### Training a Model

```python
from preprocessing import DataPreprocessor
from model import MalwareDetectionModel

# Preprocess
preprocessor = DataPreprocessor()
X, y, feature_names, df = preprocessor.preprocess("dataset.csv")

# Create and train model
model = MalwareDetectionModel(model_type='dnn')
X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2)
model.create_model(X_train.shape[1])
model.train(epochs=50, batch_size=32)

# Visualize results
model.plot_confusion_matrix(save_path="confusion_matrix.png")
model.plot_roc_curve(save_path="roc_curve.png")

# Save model
model.save_model("malware_model.h5")
```

### Generating Explanations

```python
from model import MalwareDetectionModel
from explain import SHAPExplainer

# Assume model is already trained
model = MalwareDetectionModel()
model.load_model("malware_model.h5")

# Create explainer
explainer = SHAPExplainer(model.model, feature_names)
explainer.create_explainer(X_train)

# Get feature importance
importance_df = explainer.get_feature_importance(X_test)
print(importance_df.head(10))

# Explain single prediction
explanation, fig = explainer.explain_single_instance(X_test.iloc[0])
```

## 📈 Dataset Format

Your CSV file should have:
- **At least 2 columns**: Features and one label column
- **Label column** named one of: `Class`, `Label`, `Target`, `Malware`, `Type`, `Category`
- **Label values** should indicate benign/malware samples
  - Benign: "Benign", "Normal", "Clean", "Legitimate", "0"
  - Malware: "Malware", "Malicious", "Spam", "Trojan", "1"

Example structure:
```
Feature1,Feature2,Feature3,...,FeatureN,Class
10.5,20.3,15.2,...,5.1,Benign
12.3,25.1,18.4,...,6.2,Malware
...
```

## 🎓 Expected Results

When trained on CIC-MalMem-2022 dataset with DNN:
- **Accuracy**: 0.95-0.98
- **Precision**: 0.93-0.97
- **Recall**: 0.92-0.96
- **F1-Score**: 0.92-0.96
- **ROC-AUC**: 0.96-0.99

*Results may vary depending on preprocessing and hyperparameters*

## 🛠️ Troubleshooting

### Issue: Module not found errors
```
Solution: Ensure virtual environment is activated and all packages installed
pip install -r requirements.txt --upgrade
```

### Issue: TensorFlow GPU not detected
```
Solution: This is normal. CPU mode will work fine for datasets up to 1M rows
For GPU support, install tensorflow-gpu separately
```

### Issue: "Label column not detected"
```
Solution: The CSV must have a label column with name like:
Class, Label, Target, Malware, Type, Category, or similar
Check your CSV column names
```

### Issue: Memory error during training
```
Solution: 
- Reduce batch_size in app (try 16 or 8)
- Reduce epochs
- Use a sample of your dataset first
- Reduce hidden_layers in model.py
```

### Issue: Streamlit won't start
```
Solution:
cd to project directory
streamlit run app.py --logger.level=debug
Check for Python path issues
```

## 📝 File Descriptions

### `preprocessing.py`
Handles all data operations:
- CSV loading with validation
- Automatic label column detection
- Binary label conversion
- Missing value imputation
- MinMaxScaler normalization

### `model.py`
Deep learning model implementation:
- DNN architecture (4 dense layers with batch norm)
- CNN architecture (2 conv layers with pooling)
- Train-test splitting with stratification
- Metrics calculation
- Visualization methods

### `explain.py`
SHAP-based explainability:
- KernelExplainer for model-agnostic explanations
- Global feature importance
- Instance-level explanations
- Summary and dependence plots

### `app.py`
Streamlit web application:
- Interactive UI
- File upload handling
- Real-time predictions
- Dashboard visualizations
- Session state management

### `requirements.txt`
Python package dependencies with pinned versions

## 💻 VS Code Integration

### Open Project in VS Code

```powershell
# Navigate to project
cd "e:\research code\malware-detection-xai"

# Open in VS Code
code .
```

### Configure Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the virtual environment (`venv`)
4. Verify in console: `Python 3.x.x ('./venv': venv)`

### Run Streamlit from VS Code Terminal

1. Open terminal: `` Ctrl+` ``
2. Activate venv: `.\venv\Scripts\Activate.ps1`
3. Run: `streamlit run app.py`

### Debugging

For debugging individual modules, create a `debug.py`:

```python
from preprocessing import DataPreprocessor
from model import MalwareDetectionModel

# Your debug code here
preprocessor = DataPreprocessor()
X, y, _, _ = preprocessor.preprocess("your_dataset.csv")
```

Then press `F5` to debug in VS Code.

## 🔐 Security Notes

- This tool is for authorized security research only
- Use only on datasets you own or have permission to analyze
- The model predictions should be validated by security experts
- SHAP explanations are approximate and should be treated as guidance

## 📚 References

- **SHAP**: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- **TensorFlow**: [https://tensorflow.org](https://tensorflow.org)
- **Streamlit**: [https://streamlit.io](https://streamlit.io)
- **CIC-MalMem-2022**: [https://www.unb.ca/cic/datasets/malmem2022.html](https://www.unb.ca/cic/datasets/malmem2022.html)

## 📄 License

This project is provided as-is for educational and research purposes.

## ✉️ Questions & Support

For issues:
1. Check the Troubleshooting section above
2. Review error messages in the terminal
3. Verify CSV format matches requirements
4. Try with a smaller dataset first

---

**Version**: 1.0  
**Last Updated**: April 2026  
**Status**: Ready for Production
