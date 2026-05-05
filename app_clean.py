"""
Streamlit App: Malware Detection using Explainable AI
Clean, professional UI for research presentation and demo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
import time

# Import custom modules
from preprocessing import DataPreprocessor
from model import MalwareDetectionModel
from shap_explainer import SHAPExplainer


# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Malware Detection XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        font-size: 2.2em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #555;
        font-size: 0.95em;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# ============= SESSION STATE =============
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'model': None,
        'preprocessor': None,
        'feature_names': None,
        'X_test_scaled': None,
        'y_test': None,
        'metrics': None,
        'model_history': None,
        'y_pred': None,
        'y_pred_proba': None,
        'X_train_scaled': None,
        'shap_explainer': None,
        'training_complete': False,
        'last_training_time': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()


# ============= UTILITY FUNCTIONS =============
def save_uploaded_files_to_temp(uploaded_files) -> list:
    """Save uploaded files to temporary directory."""
    temp_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_paths.append(tmp_file.name)
    return temp_paths


def cleanup_temp_files(file_paths: list):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            os.unlink(file_path)
        except:
            pass


# ============= SIDEBAR NAVIGATION =============
with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Select Mode:",
        ["Train Model", "Make Prediction", "Model Analysis"]
    )


# ============= MAIN APP =============
def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-title">🛡️ Malware Detection using Explainable AI</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">RAM-based malware detection with interpretable deep learning</div>', 
                unsafe_allow_html=True)
    
    st.divider()
    
    # ===== MODE 1: TRAIN MODEL =====
    if mode == "Train Model":
        st.header("Train Model")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload CSV file(s):",
            type=['csv'],
            accept_multiple_files=True,
            help="Select 1 or more CSV files. They will be automatically merged."
        )
        
        if uploaded_files and len(uploaded_files) > 0:
            st.success(f"✓ {len(uploaded_files)} file(s) selected")
            
            # Training configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.number_input("Epochs:", 10, 200, 100)
            with col2:
                batch_size = st.selectbox("Batch Size:", [16, 32, 64])
            with col3:
                do_cv = st.checkbox("k-Fold CV", True)
            
            # Train button
            if st.button("🚀 Train Model", use_container_width=True, type="primary"):
                try:
                    progress_bar = st.progress(0)
                    status = st.status("Initializing...", state="running")
                    
                    temp_paths = save_uploaded_files_to_temp(uploaded_files)
                    
                    try:
                        # Preprocessing
                        with status:
                            st.write("Preprocessing data...")
                        progress_bar.progress(25)
                        
                        preprocessor = DataPreprocessor()
                        X_train_scaled, X_test_scaled, y_train, y_test, feature_names = preprocessor.preprocess(temp_paths)
                        
                        # Model building
                        with status:
                            st.write("Building model...")
                        progress_bar.progress(50)
                        
                        model_obj = MalwareDetectionModel(model_type='dnn')
                        model_obj.build_dnn_model(X_train_scaled.shape[1])
                        
                        # Training
                        with status:
                            st.write("Training model...")
                        progress_bar.progress(75)
                        
                        model_obj.train(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                        
                        # Evaluation
                        with status:
                            st.write("Evaluating model...")
                        progress_bar.progress(90)
                        
                        metrics = model_obj.evaluate_on_test_set(X_test_scaled, y_test)
                        
                        # Store in session state
                        st.session_state['model'] = model_obj
                        st.session_state['preprocessor'] = preprocessor
                        st.session_state['feature_names'] = feature_names
                        st.session_state['X_test_scaled'] = X_test_scaled
                        st.session_state['y_test'] = y_test
                        st.session_state['X_train_scaled'] = X_train_scaled
                        st.session_state['metrics'] = metrics
                        st.session_state['model_history'] = model_obj.history
                        st.session_state['y_pred'] = model_obj.y_pred
                        st.session_state['y_pred_proba'] = model_obj.y_pred_proba
                        st.session_state['training_complete'] = True
                        st.session_state['last_training_time'] = time.time()
                        
                        # Initialize SHAP
                        try:
                            explainer = SHAPExplainer(model_obj.model, feature_names)
                            explainer.init_with_background_data(X_train_scaled)
                            st.session_state['shap_explainer'] = explainer
                        except Exception as e:
                            st.warning(f"SHAP initialization: {str(e)}")
                            st.session_state['shap_explainer'] = None
                        
                        progress_bar.progress(100)
                        status.update(label="✓ Training Complete!", state="complete")
                        
                        # k-fold CV
                        if do_cv:
                            st.info("Running cross-validation...")
                            try:
                                model_obj.cross_validate(X_train_scaled, y_train)
                                st.success("✓ Cross-validation completed")
                            except Exception as e:
                                st.warning(f"Cross-validation: {e}")
                        
                    finally:
                        cleanup_temp_files(temp_paths)
                
                except Exception as e:
                    st.error(f"Training Error: {str(e)}")
        
        # Display results if trained
        if st.session_state['training_complete'] and st.session_state['metrics'] is not None:
            st.divider()
            metrics = st.session_state['metrics']
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
            with col5:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            
            # Plots
            st.subheader("Visualizations")
            col1, col2, col3 = st.columns(3)
            
            model_obj = st.session_state['model']
            with col1:
                try:
                    if model_obj.history is not None:
                        fig = model_obj.plot_training_history()
                        if fig:
                            st.pyplot(fig)
                except:
                    pass
            
            with col2:
                try:
                    if model_obj.y_pred is not None:
                        fig = model_obj.plot_confusion_matrix()
                        if fig:
                            st.pyplot(fig)
                except:
                    pass
            
            with col3:
                try:
                    if model_obj.y_pred_proba is not None:
                        fig = model_obj.plot_roc_curve()
                        if fig:
                            st.pyplot(fig)
                except:
                    pass
    
    
    # ===== MODE 2: MAKE PREDICTION =====
    elif mode == "Make Prediction":
        st.header("Make Prediction")
        
        if not st.session_state['training_complete'] or st.session_state['model'] is None:
            st.warning("No trained model available.")
            st.info("👈 Train a model in the **Train Model** tab first.")
        else:
            st.success(f"✓ Model loaded with {len(st.session_state['feature_names'])} features")
            
            st.divider()
            
            # Input form
            with st.form("prediction_form"):
                st.write("Enter normalized feature values (0.0 - 1.0)")
                
                input_values = {}
                cols = st.columns(3)
                
                for idx, feature in enumerate(st.session_state['feature_names']):
                    with cols[idx % 3]:
                        input_values[feature] = st.number_input(
                            f"{feature}",
                            value=0.5,
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01
                        )
                
                predict_btn = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")
            
            # Execute prediction
            if predict_btn:
                try:
                    model_obj = st.session_state['model']
                    
                    input_array = np.array([input_values[f] for f in st.session_state['feature_names']])
                    input_array = input_array.reshape(1, -1)
                    
                    # Validate input
                    if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
                        st.error("Invalid input values")
                    else:
                        with st.spinner("Computing prediction..."):
                            pred_proba = model_obj.model.predict(input_array, verbose=0)
                            
                            if pred_proba is None or len(pred_proba) == 0:
                                st.error("Prediction failed")
                            else:
                                pred_class = 1 if pred_proba[0][0] > 0.5 else 0
                                confidence = max(pred_proba[0][0], 1 - pred_proba[0][0])
                                
                                st.divider()
                                
                                # Prediction result
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.subheader("Prediction")
                                    if pred_class == 0:
                                        st.success(f"✅ BENIGN\nConfidence: {confidence*100:.1f}%")
                                    else:
                                        st.error(f"⚠️ MALWARE\nConfidence: {confidence*100:.1f}%")
                                
                                # SHAP explanation
                                with col2:
                                    st.subheader("Why This Prediction?")
                                    
                                    explain_btn = st.button("📈 Explain with SHAP", use_container_width=True)
                                    
                                    if explain_btn:
                                        try:
                                            explainer = st.session_state.get('shap_explainer')
                                            
                                            if explainer is None:
                                                st.error("SHAP explainer not available")
                                            else:
                                                with st.spinner("Computing SHAP explanation..."):
                                                    try:
                                                        shap_exp = explainer.explain_instance(input_array, num_samples=50)
                                                        
                                                        top_features_df = explainer.get_top_contributing_features(shap_exp, top_n=5)
                                                        
                                                        st.divider()
                                                        st.write("**Top Contributing Features:**")
                                                        
                                                        display_df = pd.DataFrame({
                                                            'Feature': top_features_df['Feature'].values,
                                                            'SHAP Value': top_features_df['Impact'].values,
                                                            'Direction': top_features_df['Direction'].values,
                                                            'Importance': top_features_df['Magnitude'].values
                                                        })
                                                        
                                                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                                                        
                                                        # Waterfall plot
                                                        st.write("**Feature Contributions:**")
                                                        fig = explainer.plot_waterfall(shap_exp)
                                                        st.pyplot(fig, use_container_width=True)
                                                        
                                                        st.success("✅ Explanation complete")
                                                        
                                                    except Exception as e:
                                                        st.error(f"SHAP computation failed: {str(e)}")
                                        
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")
                
                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
    
    
    # ===== MODE 3: MODEL ANALYSIS =====
    elif mode == "Model Analysis":
        st.header("Model Analysis")
        
        if not st.session_state['training_complete'] or st.session_state['metrics'] is None:
            st.warning("No trained model available.")
            st.info("👈 Train a model in the **Train Model** tab first.")
        else:
            metrics = st.session_state['metrics']
            
            # Metrics summary
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
            with col5:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            
            st.divider()
            
            # Training history
            if st.session_state['model_history'] is not None:
                st.subheader("Training History")
                with st.expander("View Details"):
                    try:
                        model_obj = st.session_state['model']
                        if model_obj.history is not None:
                            history_df = pd.DataFrame(model_obj.history.history)
                            st.dataframe(history_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display history: {str(e)}")
            
            st.divider()
            
            # SHAP Global Explanation
            st.subheader("Explainable AI - SHAP Feature Importance")
            
            try:
                explainer = st.session_state.get('shap_explainer')
                X_test = st.session_state.get('X_test_scaled')
                
                if explainer is not None and X_test is not None:
                    if st.button("📊 Compute SHAP Explanation", use_container_width=True, type="primary"):
                        with st.spinner("Computing SHAP values (this takes 30-60 seconds)..."):
                            try:
                                shap_result = explainer.explain_batch(X_test, max_samples=10, num_samples=100)
                                
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.write("**Top 10 Features**")
                                    importance_df = explainer.get_feature_importance(shap_result, top_n=10)
                                    st.dataframe(importance_df, use_container_width=True)
                                
                                with col2:
                                    st.write("**Feature Importance Plot**")
                                    fig = explainer.plot_summary(shap_result, plot_type='bar', max_display=10)
                                    st.pyplot(fig)
                                
                                st.success("✅ SHAP computation complete")
                                
                            except Exception as e:
                                st.error(f"SHAP Error: {str(e)}")
                elif explainer is None:
                    st.warning("SHAP explainer not initialized")
                else:
                    st.warning("Test data not available")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
