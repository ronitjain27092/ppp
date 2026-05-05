"""
Streamlit Web Application for Malware Detection with Explainable AI
- Upload dataset
- Train/Load model
- Make predictions
- Display SHAP explanations
- Visualize results
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

# Import custom modules
from preprocessing import DataPreprocessor
from model import MalwareDetectionModel
from explain import SHAPExplainer


# Streamlit page configuration
st.set_page_config(
    page_title="Malware Detection XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        color: #FF6B6B;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #666;
        font-size: 1.1em;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .benign-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
    .malware-box {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-title">🛡️ Malware Detection with Explainable AI</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">RAM Forensics Analysis using Deep Learning & SHAP</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio(
            "Select Mode:",
            ["📊 Train Model", "🔍 Make Prediction", "📈 Model Analysis"]
        )
    
    # Mode 1: Train Model
    if mode == "📊 Train Model":
        st.header("Train New Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Step 1: Upload Dataset")
            uploaded_file = st.file_uploader(
                "Upload CIC-MalMem-2022 CSV file",
                type=['csv'],
                help="CSV file with features and a label column (Class, Label, etc.)"
            )
        
        with col2:
            st.subheader("Configuration")
            model_type = st.selectbox("Model Type:", ["dnn", "cnn"])
            test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
            epochs = st.slider("Epochs:", 10, 100, 50)
        
        if uploaded_file is not None:
            try:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                st.info("✓ File uploaded successfully")
                
                # Preprocessing
                if st.button("🚀 Start Training", use_container_width=True):
                    with st.spinner("⏳ Preprocessing data..."):
                        preprocessor = DataPreprocessor()
                        X, y, feature_names, df = preprocessor.preprocess(tmp_path)
                        
                        st.session_state.preprocessor = preprocessor
                        st.session_state.feature_names = feature_names
                    
                    # Split and prepare model
                    with st.spinner("⏳ Building model..."):
                        model_obj = MalwareDetectionModel(model_type=model_type)
                        X_train, X_test, y_train, y_test = model_obj.split_data(X, y, test_size=test_size)
                        model_obj.create_model(X_train.shape[1])
                        
                        st.session_state.model = model_obj
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                    
                    # Train
                    with st.spinner("⏳ Training model (this may take a while)..."):
                        model_obj.train(epochs=epochs, verbose=0)
                    
                    st.success("✓ Model trained successfully!")
                    
                    # Display metrics
                    st.subheader("📊 Model Performance")
                    
                    metrics = model_obj.metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1']:.4f}")
                    with col5:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    
                    # Visualizations
                    st.subheader("📈 Visualizations")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info("Training History")
                        fig = model_obj.plot_training_history()
                        st.pyplot(fig)
                    
                    with col2:
                        st.info("Confusion Matrix")
                        fig = model_obj.plot_confusion_matrix()
                        st.pyplot(fig)
                    
                    with col3:
                        st.info("ROC Curve")
                        fig = model_obj.plot_roc_curve()
                        st.pyplot(fig)
                    
                    # Create SHAP explainer
                    with st.spinner("⏳ Creating SHAP explainer..."):
                        explainer = SHAPExplainer(model_obj.model, feature_names)
                        explainer.create_explainer(X_train, sample_size=100)
                        st.session_state.explainer = explainer
                    
                    st.success("✓ Explainer created!")
                
                # Cleanup
                os.unlink(tmp_path)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your dataset format and try again.")
    
    # Mode 2: Make Prediction
    elif mode == "🔍 Make Prediction":
        st.header("Make Prediction on New Sample")
        
        if st.session_state.model is None:
            st.warning("⚠️ No trained model available. Please train a model first.")
            st.info("Go to '📊 Train Model' tab to train a new model.")
        else:
            st.subheader("Enter or Upload Data")
            
            input_method = st.radio("Input Method:", ["Manual Input", "Upload File"])
            
            if input_method == "Manual Input":
                st.info(f"Features: {len(st.session_state.feature_names)}")
                
                # Create input fields for each feature
                input_values = {}
                cols = st.columns(3)
                
                for idx, feature in enumerate(st.session_state.feature_names):
                    with cols[idx % 3]:
                        input_values[feature] = st.number_input(
                            f"{feature}",
                            value=0.5,
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            help="Value should be normalized between 0 and 1"
                        )
                
                if st.button("🔮 Predict", use_container_width=True):
                    # Prepare input
                    input_array = np.array([input_values[f] for f in st.session_state.feature_names])
                    input_array = input_array.reshape(1, -1)
                    
                    # Make prediction
                    with st.spinner("⏳ Making prediction..."):
                        if st.session_state.model.model_type == 'cnn':
                            input_array_reshaped = input_array.reshape(input_array.shape[0], input_array.shape[1], 1)
                        else:
                            input_array_reshaped = input_array
                        
                        pred_proba = st.session_state.model.model.predict(input_array_reshaped, verbose=0)
                        pred_class = 1 if pred_proba[0][0] > 0.5 else 0
                        confidence = max(pred_proba[0][0], 1 - pred_proba[0][0])
                    
                    # Display prediction
                    st.subheader("🎯 Prediction Result")
                    
                    if pred_class == 0:
                        st.markdown(
                            f'<div class="benign-box"><h2>✅ BENIGN</h2><h3>Confidence: {confidence*100:.2f}%</h3></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="malware-box"><h2>⚠️ MALWARE DETECTED</h2><h3>Confidence: {confidence*100:.2f}%</h3></div>',
                            unsafe_allow_html=True
                        )
                    
                    # SHAP explanation
                    if st.session_state.explainer is not None:
                        st.subheader("🔍 SHAP Explanation")
                        
                        with st.spinner("⏳ Generating SHAP explanation..."):
                            explanation, fig = st.session_state.explainer.explain_single_instance(input_array)
                        
                        st.pyplot(fig)
                        
                        # Show top contributing features
                        st.subheader("📊 Top Contributing Features")
                        
                        feature_contrib = pd.DataFrame({
                            'Feature': st.session_state.feature_names,
                            'Value': input_array[0],
                            'SHAP Value': explanation['shap_values']
                        }).sort_values('SHAP Value', ascending=False, key=abs)
                        
                        st.dataframe(feature_contrib.head(10), use_container_width=True)
            
            elif input_method == "Upload File":
                uploaded_file = st.file_uploader("Upload CSV with samples", type=['csv'])
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        
                        # Select columns matching training features
                        available_cols = [col for col in st.session_state.feature_names if col in df.columns]
                        
                        if len(available_cols) < len(st.session_state.feature_names):
                            st.warning(f"Only {len(available_cols)} of {len(st.session_state.feature_names)} features found")
                        
                        X_input = df[available_cols].values
                        
                        if st.button("🔮 Predict All Samples", use_container_width=True):
                            with st.spinner("⏳ Making predictions..."):
                                if st.session_state.model.model_type == 'cnn':
                                    X_reshaped = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)
                                else:
                                    X_reshaped = X_input
                                
                                pred_proba = st.session_state.model.model.predict(X_reshaped, verbose=0)
                                pred_class = (pred_proba > 0.5).astype(int).flatten()
                                confidence = np.max(np.concatenate([pred_proba, 1-pred_proba], axis=1), axis=1)
                            
                            results_df = pd.DataFrame({
                                'Prediction': ['Benign' if p == 0 else 'Malware' for p in pred_class],
                                'Confidence': confidence,
                                'Probability': pred_proba.flatten()
                            })
                            
                            st.subheader("📊 Predictions")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Samples", len(results_df))
                            with col2:
                                st.metric("Malware Detected", (pred_class == 1).sum())
                            with col3:
                                st.metric("Avg Confidence", f"{confidence.mean()*100:.2f}%")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Mode 3: Model Analysis
    elif mode == "📈 Model Analysis":
        st.header("Feature Importance Analysis")
        
        if st.session_state.model is None or st.session_state.explainer is None:
            st.warning("⚠️ No trained model available. Please train a model first.")
        else:
            st.subheader("Feature Importance (SHAP)")
            
            if st.session_state.X_test is not None:
                with st.spinner("⏳ Calculating feature importance..."):
                    importance_df = st.session_state.explainer.get_feature_importance(
                        st.session_state.X_test,
                        num_samples=min(100, len(st.session_state.X_test))
                    )
                
                st.subheader("Top Features")
                st.dataframe(importance_df.head(20), use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Importance Plot")
                    fig = st.session_state.explainer.plot_summary(
                        st.session_state.X_test,
                        num_samples=min(50, len(st.session_state.X_test))
                    )
                    st.pyplot(fig)
            
            st.subheader("Model Performance Summary")
            
            if st.session_state.model.metrics:
                metrics_df = pd.DataFrame(
                    st.session_state.model.metrics.items(),
                    columns=['Metric', 'Score']
                )
                st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    main()
