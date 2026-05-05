# Project Architecture & Data Flow

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 MALWARE DETECTION XAI SYSTEM                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                           │
│                                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  📊 Train Model  │  │  🔍 Predict      │  │  📈 Analyze      │
│  │  • Upload CSV    │  │  • Manual Input  │  │  • Feature Rank  │
│  │  • Configure     │  │  • Batch Upload  │  │  • Importance    │
│  │  • Train DNN/CNN │  │  • Get Pred      │  │  • Metrics       │
│  │  • View Results  │  │  • SHAP Explian  │  │  • Plots         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│
│                    STREAMLIT WEB INTERFACE (app.py)
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                   PROCESSING PIPELINE                             │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐
│  │ DATA PREPROCESSING MODULE (preprocessing.py)                  │
│  │                                                               │
│  │  CSV Input → Load → Detect Label → Convert Labels            │
│  │      ↓         ↓         ↓              ↓                     │
│  │  Handle Missing Values → Normalize (MinMaxScaler)            │
│  │      ↓                                                         │
│  │  Output: X (normalized features), y (binary labels)          │
│  └──────────────────────────────────────────────────────────────┘
│                              ↓
│  ┌──────────────────────────────────────────────────────────────┐
│  │ MODEL TRAINING MODULE (model.py)                             │
│  │                                                               │
│  │  X, y → Train/Test Split (80/20) → Create Model             │
│  │      ↓                              ↓                         │
│  │  Train DNN/CNN ← Early Stopping    ↓                         │
│  │      ↓                              ↓                         │
│  │  Evaluate → Metrics (Acc, Prec, Recall, F1, AUC)           │
│  │      ↓                              ↓                         │
│  │  Trained Model + Predictions ready for explanation          │
│  └──────────────────────────────────────────────────────────────┘
│                              ↓
│  ┌──────────────────────────────────────────────────────────────┐
│  │ EXPLAINABILITY MODULE (explain.py)                           │
│  │                                                               │
│  │  Trained Model → Create SHAP Explainer                       │
│  │      ↓                        ↓                               │
│  │  X_train (background)        KernelExplainer                │
│  │      ↓                        ↓                               │
│  │  Calculate SHAP Values → Feature Importance                 │
│  │      ↓                  ↓                                     │
│  │  Instance Explanations  Summary Plots                        │
│  │      ↓                  ↓                                     │
│  │  SHAP Feature Contributions visualized                       │
│  └──────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                  │
│                                                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Metrics         │  │ Visualizations  │  │ Predictions     │  │
│  │ • Accuracy      │  │ • Confusion Mtx │  │ • Class Label   │  │
│  │ • Precision     │  │ • ROC Curve     │  │ • Confidence    │  │
│  │ • Recall        │  │ • Training Plot │  │ • Probabilities │  │
│  │ • F1-Score      │  │ • SHAP Plots    │  │ • Explanations  │  │
│  │ • ROC-AUC       │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

```
INPUT CSV
    │
    ├─→ Read CSV File
    │       │
    │       ├─→ Detect Label Column
    │       │       │
    │       │       └─→ "Class"? "Label"? "Target"?
    │       │
    │       ├─→ Convert Labels to Binary (0/1)
    │       │       │
    │       │       └─→ Benign=0, Malware=1
    │       │
    │       ├─→ Handle Missing Values
    │       │       │
    │       │       ├─→ Numeric: fill with median
    │       │       └─→ Categorical: fill with mode
    │       │
    │       └─→ Normalize Features (MinMaxScaler)
    │               │
    │               └─→ Scale to [0, 1] range
    │
    ├─→ NORMALIZED DATA (X, y)
    │
    ├─→ Train/Test Split (80/20)
    │       │
    │       ├─→ X_train, y_train (80%)
    │       └─→ X_test, y_test (20%)
    │
    ├─→ CREATE MODEL
    │       │
    │       ├─→ DNN: Dense(128)→BN→Dropout → Dense(64)→...
    │       │
    │       └─→ CNN: Conv1D(64) → MaxPool → Conv1D(32) → Flatten → Dense
    │
    ├─→ TRAIN MODEL
    │       │
    │       ├─→ Fit on X_train with validation
    │       ├─→ Early stopping if val_loss plateaus
    │       └─→ Save weights and history
    │
    ├─→ EVALUATE MODEL
    │       │
    │       ├─→ Predictions on X_test
    │       ├─→ Calculate Metrics
    │       │       └─→ Accuracy, Precision, Recall, F1, AUC
    │       │
    │       └─→ Generate Plots
    │               ├─→ Confusion Matrix
    │               ├─→ ROC Curve
    │               └─→ Training History
    │
    ├─→ CREATE SHAP EXPLAINER
    │       │
    │       ├─→ Background data: X_train sample
    │       ├─→ Model: trained neural network
    │       └─→ Method: KernelExplainer
    │
    ├─→ GENERATE EXPLANATIONS
    │       │
    │       ├─→ Global Feature Importance
    │       │       └─→ Top features across all predictions
    │       │
    │       ├─→ Instance Explanations
    │       │       └─→ Why this sample is malware/benign
    │       │
    │       └─→ Visualizations
    │               ├─→ Summary Plots
    │               ├─→ Dependence Plots
    │               └─→ Instance Force Plots
    │
    └─→ OUTPUT (Predictions + Explanations)
            │
            ├─→ Prediction: "Malware" or "Benign"
            ├─→ Confidence: 0.95 (95%)
            └─→ SHAP: Feature contributions to decision
```

---

## 🔄 Module Dependencies

```
app.py (Streamlit Web Interface)
    │
    ├── from preprocessing import DataPreprocessor
    │       └── pandas, numpy, sklearn
    │
    ├── from model import MalwareDetectionModel
    │       └── tensorflow, keras, sklearn, numpy, matplotlib, seaborn
    │
    ├── from explain import SHAPExplainer
    │       └── shap, numpy, pandas, matplotlib
    │
    └── External: streamlit, matplotlib


preprocessing.py (Independent)
    ├── pandas
    ├── numpy
    ├── sklearn.preprocessing (MinMaxScaler)
    └── os


model.py (Depends on preprocessing)
    ├── numpy
    ├── pandas
    ├── tensorflow, keras
    ├── sklearn.model_selection
    ├── sklearn.metrics
    ├── matplotlib
    └── seaborn


explain.py (Depends on model)
    ├── numpy
    ├── pandas
    ├── matplotlib
    └── shap


test_installation.py (Tests all modules)
    ├── All of the above
    └── Generates sample data
```

---

## 🎯 Training Pipeline Details

```
┌──────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                           │
└──────────────────────────────────────────────────────────────┘

INPUT: X_train (samples × features), y_train (binary labels)

┌─────────────────────────────────────────────────────────────┐
│ DNN ARCHITECTURE                                             │
├─────────────────────────────────────────────────────────────┤
│ Layer 1:  Input (features) → Dense(128) → ReLU             │
│           └→ BatchNorm → Dropout(0.3)                      │
│                                                              │
│ Layer 2:  Dense(64) → ReLU → BatchNorm → Dropout(0.3)     │
│                                                              │
│ Layer 3:  Dense(32) → ReLU → BatchNorm → Dropout(0.2)     │
│                                                              │
│ Layer 4:  Dense(16) → ReLU → Dropout(0.2)                 │
│                                                              │
│ Output:   Dense(1) → Sigmoid → Probability [0, 1]         │
└─────────────────────────────────────────────────────────────┘

OPTIMIZER: Adam (learning_rate=0.001)
LOSS: Binary Crossentropy
METRICS: Accuracy, AUC

┌─────────────────────────────────────────────────────────────┐
│ PER EPOCH (50 epochs)                                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Forward pass: compute predictions
│ 2. Calculate loss on training data
│ 3. Backward pass: compute gradients
│ 4. Update weights: W_new = W_old - lr × gradient
│ 5. Validate on 20% validation split
│ 6. Early stopping: if val_loss doesn't improve for 10 epochs
│
│ Output: loss, accuracy, validation_loss, validation_accuracy
└─────────────────────────────────────────────────────────────┘

EVALUATION: test_accuracy, test_precision, test_recall, etc.

OUTPUT: Trained model weights + metrics + predictions
```

---

## 📈 SHAP Explanation Process

```
┌──────────────────────────────────────────────────────────┐
│              SHAP EXPLAINABILITY PROCESS                  │
└──────────────────────────────────────────────────────────┘

Input: Trained Model + Test Data

┌─────────────────────────────────────────────────────────┐
│ 1. CREATE EXPLAINER                                     │
├─────────────────────────────────────────────────────────┤
│    Background Data (sample of X_train)
│              ↓
│    KernelExplainer Initialization
│              ↓
│    Store expected value (base prediction)
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 2. CALCULATE SHAP VALUES (Per Instance)                 │
├─────────────────────────────────────────────────────────┤
│    Sample instance: [0.5, 0.3, 0.8, ..., 0.2]
│              ↓
│    Perturb each feature, measure impact
│              ↓
│    Use Shapley values from game theory
│              ↓
│    Output: SHAP values per feature
│    Example: [0.2, -0.1, 0.15, ..., 0.05]
│
│    Interpretation: positive = increases malware score
│                   negative = decreases malware score
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 3. VISUALIZATIONS                                       │
├─────────────────────────────────────────────────────────┤
│    a) Feature Importance (Mean |SHAP|)
│       Feature_A: 0.3 ████████    [Most important]
│       Feature_B: 0.2 █████
│       Feature_C: 0.1 ███         [Least important]
│
│    b) Summary Plot (Beeswarm)
│       Shows distribution of SHAP values per feature
│       Color: blue=decreases, red=increases
│
│    c) Instance Plot (Waterfall)
│       Base value: 0.4
│         ↓
│       Feature_A: +0.2 → 0.6
│       Feature_B: -0.1 → 0.5
│       Feature_C: +0.1 → 0.6
│         ↓
│       Final: 0.95 (high malware probability)
└─────────────────────────────────────────────────────────┘

OUTPUT: Visual explanations + feature importance rankings
```

---

## 🔐 Prediction Pipeline

```
NEW SAMPLE INPUT
    │
    ├─→ Load trained model
    │
    ├─→ Normalize input (using fitted scaler)
    │
    ├─→ MODEL INFERENCE
    │       │
    │       └─→ Forward pass through DNN/CNN
    │           └─→ Output: probability [0, 1]
    │               ├─→ 0.3 = 30% malware = BENIGN
    │               └─→ 0.7 = 70% malware = MALWARE
    │
    ├─→ EXPLAINABILITY
    │       │
    │       └─→ SHAP explainer
    │           ├─→ Calculate SHAP values
    │           └─→ Generate explanation
    │               ├─→ Which features pushed towards malware?
    │               ├─→ Which features pushed towards benign?
    │               └─→ Feature contribution magnitudes
    │
    └─→ OUTPUT
        ├─→ Prediction: "Malware" or "Benign"
        ├─→ Confidence: 95%
        ├─→ Probability: 0.95
        └─→ SHAP Explanation: [Feature contributions]
            Example:
            Feature_A (0.8): +0.25 (supports Malware)
            Feature_B (0.2): -0.15 (supports Benign)
            Feature_C (0.6): +0.10 (supports Malware)
            ...
            FINAL: Malware (95% confidence)
```

---

## 💾 File I/O Flow

```
INPUT FILES:
    CSV Dataset (your data)
    └─→ app.py: Upload via Streamlit
    └─→ preprocessing.py: Load & preprocess
    
INTERMEDIATE FILES (Generated):
    sample_dataset.csv (from test_installation.py)
    
OUTPUT FILES (Generated by app):
    trained_model.h5 (optional save)
    confusion_matrix.png
    roc_curve.png
    training_history.png
    feature_importance.csv
    
IN-MEMORY DATA:
    X, y (preprocessed features and labels)
    Model weights
    SHAP values
    Predictions & confidences
```

---

## 🔌 Integration Points

If you want to integrate this into your own system:

```python
# Import the modules
from preprocessing import DataPreprocessor
from model import MalwareDetectionModel
from explain import SHAPExplainer

# Step 1: Preprocess your data
preprocessor = DataPreprocessor()
X, y, features, df = preprocessor.preprocess("your_data.csv")

# Step 2: Train model (or load pretrained)
model = MalwareDetectionModel()
model.load_model("pretrained_model.h5")

# Step 3: Make predictions
prediction_proba = model.model.predict(X_new)
prediction = "Malware" if prediction_proba > 0.5 else "Benign"

# Step 4: Explain prediction
explainer = SHAPExplainer(model.model, features)
explanation, fig = explainer.explain_single_instance(X_new)

# Use in your system:
# - Log predictions to database
# - Trigger alerts for malware
# - Generate security reports
# - Feed to other systems
```

---

This architecture is designed for:
- ✅ Easy to use (Streamlit interface)
- ✅ Explainable (SHAP integration)
- ✅ Scalable (batch predictions)
- ✅ Maintainable (modular code)
- ✅ Professional (production-ready)
