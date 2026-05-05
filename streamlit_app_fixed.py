"""
Streamlit Web App - CNN Malware Detection

Web interface for training and analyzing CNN malware detection models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Custom modules
from preprocessing import DataPreprocessor
from cnn_model_fixed import CNNMalwareDetector
from evaluation_fixed import ModelEvaluator, find_optimal_threshold

# Import SHAP explainer module
try:
    from shap_module import SHAPExplainer, explain_model, display_global_explanation, display_local_explanation
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Streamlit page config
st.set_page_config(
    page_title="CNN Malware Detection (Class Imbalance Fixed)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .header-title {
        color: #FF6B6B;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'evaluator_results' not in st.session_state:
    st.session_state.evaluator_results = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.3
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None
if 'X_train_scaled' not in st.session_state:
    st.session_state.X_train_scaled = None
if 'X_test_scaled' not in st.session_state:
    st.session_state.X_test_scaled = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'shap_values_dict' not in st.session_state:
    st.session_state.shap_values_dict = None


def main():
    """Main app."""
    
    # Header
    st.markdown('<div class="header-title">🛡️ CNN Malware Detection</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio(
            "Select Mode:",
            ["📊 Train Model", "🔍 Analyze Results", "🧠 Explainable AI"]
        )
        
        st.markdown("---")
        st.subheader("Model Parameters")
        epochs = st.slider("Epochs:", 5, 50, 20)
        batch_size = st.selectbox("Batch Size:", [16, 32, 64])
        threshold = st.slider("Prediction Threshold:", 0.1, 0.9, 0.3, 0.05)
        st.session_state.threshold = threshold
    
    # Mode 1: Train Model
    if mode == "📊 Train Model":
        st.header("Train CNN Model")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "📁 Upload CSV file(s):",
            type=['csv'],
            accept_multiple_files=True,
            help="CIC-MalMem dataset"
        )
        
        if uploaded_files and len(uploaded_files) > 0:
            st.success(f"✓ {len(uploaded_files)} file(s) selected")
            
            if st.button("🚀 START TRAINING", use_container_width=True):
                try:
                    # Save files
                    temp_paths = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                            tmp.write(file.getbuffer())
                            temp_paths.append(tmp.name)
                    
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_placeholder = st.empty()
                        
                        # Step 1: Preprocessing
                        status_placeholder.write("⏳ Step 1: Preprocessing data...")
                        progress_bar.progress(20)
                        
                        preprocessor = DataPreprocessor()
                        X_train_scaled, X_test_scaled, y_train, y_test, feature_names = \
                            preprocessor.preprocess(temp_paths if len(temp_paths) > 1 else temp_paths[0])
                        
                        st.session_state.preprocessor = preprocessor
                        st.session_state.feature_names = feature_names
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Train Samples", len(X_train_scaled))
                        with col2:
                            st.metric("Test Samples", len(X_test_scaled))
                        with col3:
                            st.metric("Features", X_train_scaled.shape[1])
                        
                        # Step 2: Build Model
                        status_placeholder.write("⏳ Step 2: Building CNN model...")
                        progress_bar.progress(40)
                        
                        model = CNNMalwareDetector(input_dim=X_train_scaled.shape[1])
                        model.build_cnn_model()
                        
                        st.session_state.model = model
                        
                        # Step 3: Compute Class Weights
                        status_placeholder.write("⏳ Step 3: Computing class weights...")
                        progress_bar.progress(50)
                        
                        model.compute_class_weights(y_train)
                        
                        # Step 4: Train
                        status_placeholder.write(f"⏳ Step 4: Training model ({epochs} epochs)...")
                        progress_bar.progress(60)
                        
                        model.train(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                        
                        # Step 5: Evaluate
                        status_placeholder.write("⏳ Step 5: Evaluating on test set...")
                        progress_bar.progress(80)
                        
                        # Evaluate with custom threshold
                        metrics = model.evaluate(X_test_scaled, y_test, threshold=threshold)
                        
                        # Step 6: Analysis
                        status_placeholder.write("⏳ Step 6: Analyzing predictions...")
                        progress_bar.progress(90)
                        
                        evaluator_results = ModelEvaluator.analyse_predictions(
                            y_test, model.y_pred, model.y_pred_proba,
                            model_name="CNN", threshold=threshold
                        )
                        st.session_state.evaluator_results = evaluator_results
                        
                        # Step 7: Initialize SHAP Explainer
                        status_placeholder.write("⏳ Step 7: Initializing SHAP explainer...")
                        progress_bar.progress(95)
                        
                        try:
                            # Store data for SHAP
                            st.session_state.X_train_scaled = X_train_scaled
                            st.session_state.X_test_scaled = X_test_scaled
                            st.session_state.y_test = y_test
                            
                            if SHAP_AVAILABLE:
                                # Initialize SHAP explainer with training data (background)
                                shap_explainer = explain_model(
                                    model=model.model,
                                    X_train=X_train_scaled,
                                    X_test=X_test_scaled,
                                    feature_names=feature_names
                                )
                                st.session_state.shap_explainer = shap_explainer
                                status_placeholder.write("✓ SHAP explainer ready!")
                            else:
                                st.session_state.shap_explainer = None
                        except Exception as e:
                            st.warning(f"⚠️ SHAP initialization note: {str(e)}")
                            st.session_state.shap_explainer = None
                        
                        progress_bar.progress(100)
                        status_placeholder.success("✓ Training Complete!")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        # Metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.4f}")
                        with col3:
                            st.metric("Recall ⭐", f"{metrics['recall']:.4f}")
                        with col4:
                            st.metric("F1-Score", f"{metrics['f1']:.4f}")
                        with col5:
                            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                        
                        # Validation checks
                        st.subheader("✓ Validation Checks")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if metrics['recall'] > 0:
                                st.success("✓ Recall > 0 (Malware IS being detected!)")
                            else:
                                st.error("❌ Recall = 0 (Malware NOT detected - class imbalance issue!)")
                        
                        with col2:
                            if metrics['f1'] > 0:
                                st.success("✓ F1-Score > 0 (Good balance)")
                            else:
                                st.error("❌ F1-Score = 0 (Model not detecting minority class)")
                        
                        # Visualizations
                        st.subheader("📈 Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Confusion Matrix**")
                            fig_cm = model.plot_confusion_matrix()
                            if fig_cm:
                                st.pyplot(fig_cm)
                        
                        with col2:
                            st.write("**Training History**")
                            fig_history = model.plot_training_history()
                            if fig_history:
                                st.pyplot(fig_history)
                        
                        # Probability distribution
                        st.write("**Probability Distribution**")
                        fig_prob = ModelEvaluator.plot_probability_distribution(
                            model.y_pred_proba, y_test, threshold=threshold
                        )
                        if fig_prob:
                            st.pyplot(fig_prob)
                        
                        # ROC curve
                        st.write("**ROC Curve**")
                        fig_roc = ModelEvaluator.plot_roc_curve(y_test, model.y_pred_proba)
                        if fig_roc:
                            st.pyplot(fig_roc)
                    
                    finally:
                        # Cleanup
                        for path in temp_paths:
                            try:
                                os.unlink(path)
                            except:
                                pass
                
                except Exception as e:
                    st.error(f"❌ Training Error: {str(e)}")
                    st.write("**Troubleshooting:**")
                    st.write("• Ensure CSV file has valid format")
                    st.write("• Check dataset has both Benign and Malware samples")
                    st.write("• Verify column names are correct")
    
    # Mode 2: Analyze Results
    elif mode == "🔍 Analyze Results":
        st.header("Analysis of Results")
        
        if st.session_state.model is None:
            st.warning("⚠️ No trained model available. Train a model first.")
        else:
            model = st.session_state.model
            
            # Show evaluation results
            if st.session_state.evaluator_results:
                st.subheader("Evaluation Metrics")
                results = st.session_state.evaluator_results
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Score': [
                        results['accuracy'],
                        results['precision'],
                        results['recall'],
                        results['f1']
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
    
    # Mode 3: Explainable AI
    elif mode == "🧠 Explainable AI":
        st.header("🧠 Explainable AI - SHAP Analysis")
        
        if st.session_state.model is None:
            st.warning("⚠️ No trained model available. Train a model first.")
        elif not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP not installed. Install it with: pip install shap")
        elif st.session_state.shap_explainer is None:
            st.warning("⚠️ SHAP explainer not initialized during training.")
        else:
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["🌍 Global Importance", "🔍 Local Explanation", "❓ Why SHAP?"])
            
            with tab1:
                display_global_explanation(
                    st.session_state.shap_explainer,
                    st.session_state.X_test_scaled
                )
            
            with tab2:
                display_local_explanation(
                    st.session_state.shap_explainer,
                    st.session_state.X_test_scaled,
                    st.session_state.model.y_pred,
                    st.session_state.model.y_pred_proba,
                    st.session_state.y_test
                )
            
            with tab3:
                st.markdown("""
                ### 🤔 Why is SHAP Important for Malware Detection?
                
                #### 1. **Trust Verification**
                - ✓ Verify model uses REAL malware signals (not artifacts)
                - ✓ Rule out spurious correlations
                - ✓ Security teams can validate decisions
                
                #### 2. **Debugging & Improvement**
                - 🔍 Identify why false positives/negatives occur
                - 🔍 Discover if model relies on suspicious features
                - 🔍 Example: If "timestamp" matters, that's wrong!
                
                #### 3. **Forensics Application**
                **CIC-MalMem Dataset:** RAM memory features from infected machines
                - 🧠 Features like: process handles, memory sectors, API calls
                - 🧠 SHAP shows WHICH memory patterns = malware
                - 🧠 Forensic analysts can verify predictions match real behavior
                
                #### 4. **Compliance & Legal**
                - 📋 Many regulations require interpretable AI
                - 📋 SHAP provides audit trail for automated decisions
                - 📋 Essential for security-critical systems
                
                #### 5. **Fast Investigation**
                - ⚡ When model flags suspicious file: "Why?"
                - ⚡ SHAP answer: "Top 5 features pushing malware prediction"
                - ⚡ Faster than re-analyzing entire model
                
                ---
                
                ### 📊 How to Read the Plots
                
                **Global Importance (Bar plot):**
                | Feature | Importance | Meaning |
                |---------|-----------|---------|
                | DLL Loads | ████ | Very important for malware detection |
                | Registry Ops | ███ | Important |
                | File Writes | ██ | Moderate |
                | Process Creates | █ | Low importance |
                
                **Local Explanation (Waterfall plot):**
                ```
                Base probability: 20% (neutral)
                + Feature A: +30% (pushes toward MALWARE) 🔴
                + Feature B: +25% (pushes toward MALWARE) 🔴
                - Feature C: -5% (pushes toward BENIGN) 🟢
                ─────────────────
                Final prediction: 70% MALWARE
                ```
                
                ---
                
                ### 🛠️ Technical: What's Happening
                
                **SHAP Method:** KernelExplainer (model-agnostic)
                - ✓ Works with any model (CNN, Random Forest, etc.)
                - ✓ Safe and mathematically sound
                - ✓ Based on game theory (Shapley values)
                
                **Process:**
                1. Takes background samples (100 training samples)
                2. Perturbs each feature of target sample
                3. Measures change in prediction
                4. Computes SHAP value (feature contribution)
                
                **Performance Notes:**
                - Uses first 100 training samples as background (~3s startup)
                - Global explanation: ~30s for 50 test samples
                - Local explanation: ~10s per sample
                
                """)

if __name__ == "__main__":
    main()

