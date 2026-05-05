"""
SHAP-based Explainable AI for CNN Malware Detection

This module provides SHAP (SHapley Additive exPlanations) integration
for explaining CNN model predictions on CIC-MalMem dataset.

WHY SHAP FOR MALWARE DETECTION?
================================
1. TRUST: Verify model uses real malware signals, not artifacts
2. FORENSICS: Show which RAM features indicate malicious behavior
3. DEBUGGING: Understand why false positives/negatives occur
4. COMPLIANCE: Provide interpretable decision reasoning
5. CONFIDENCE: Help security teams trust automated decisions

HOW IT WORKS:
=============
SHAP values quantify each feature's contribution to prediction:
- Positive SHAP: Feature contributes to "Malware" prediction
- Negative SHAP: Feature contributes to "Benign" prediction
- Magnitude: How important is this feature for this prediction

KERNEL EXPLAINER (CHOSEN):
==========================
- ✓ Model-agnostic (works with any model, including CNN)
- ✓ Stable and reliable
- ✓ Clear mathematical foundation (Shapley values)
- ✓ Safe with tabular data (our use case)
- ~ Slower than DeepExplainer (but acceptable for sample batches)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """Explain CNN malware detection predictions using SHAP."""
    
    def __init__(self, model, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer for CNN model.
        
        Args:
            model: Trained Keras/TensorFlow model
            X_train: Training data for background (uses subset)
            feature_names: List of feature names (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.X_background = None
        
        # Use subset of training data for background (max 100 samples for speed)
        n_background = min(100, len(X_train))
        self.X_background = X_train[:n_background]
        
        print(f"✓ SHAP Explainer initialized with {n_background} background samples")
    
    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """
        Ensure input is 2D for SHAP (remove channel dimension if added).
        
        Args:
            X: Input array (may be 2D or 3D)
            
        Returns:
            2D array suitable for SHAP
        """
        if len(X.shape) == 3:
            # If it's (samples, features, 1), squeeze to (samples, features)
            if X.shape[2] == 1:
                return X.squeeze(axis=2)
        return X
    
    def explain_global(self, X_test: np.ndarray, max_display: int = 15) -> plt.Figure:
        """
        Generate global explanation showing feature importance.
        
        Shows which features are most important across ALL predictions.
        
        Args:
            X_test: Test data to explain (uses first 50)
            max_display: Number of top features to show
            
        Returns:
            matplotlib Figure with SHAP summary plot
        """
        try:
            # Use small subset for speed
            X_subset = X_test[:min(50, len(X_test))]
            X_bg = self._prepare_input(self.X_background)
            X_subset_prepared = self._prepare_input(X_subset)
            
            # Create explainer (KernelExplainer - safe, stable)
            print(f"  Computing SHAP values for {len(X_subset_prepared)} samples...")
            explainer = shap.KernelExplainer(
                model=lambda x: self.model.predict(x, verbose=0),
                data=X_bg
            )
            
            # Get SHAP values
            shap_values = explainer.shap_values(X_subset_prepared)
            
            # Handle output (may be list or array)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Set feature names if not provided
            if self.feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_subset_prepared.shape[1])]
            else:
                feature_names = self.feature_names
            
            # Calculate mean absolute SHAP values for importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            top_values = mean_abs_shap[top_indices]
            top_names = [feature_names[int(i)] if int(i) < len(feature_names) else f"Feature_{i}" 
                        for i in top_indices]
            
            ax.barh(range(len(top_values)), top_values, color='steelblue')
            ax.set_yticks(range(len(top_values)))
            ax.set_yticklabels(top_names)
            ax.set_xlabel('Mean |SHAP value| (average absolute impact on model output)')
            ax.set_ylabel('Features')
            ax.set_title('SHAP Summary Plot - Feature Importance')
            ax.invert_yaxis()
            
            plt.tight_layout()
            print("  ✓ Global explanation generated successfully")
            return fig
            
        except Exception as e:
            print(f"  ✗ Error in global explanation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def explain_local(self, X_test: np.ndarray, sample_idx: int, 
                     model_proba: float = None, y_true: int = None) -> Tuple[plt.Figure, Dict]:
        """
        Generate local explanation for single sample.
        
        Shows which features contributed to prediction for THIS specific sample.
        
        Args:
            X_test: Test data
            sample_idx: Index of sample to explain
            model_proba: Model's probability output (optional, for display)
            y_true: True label (optional, for display)
            
        Returns:
            Tuple of (matplotlib Figure, explanation dict)
        """
        try:
            if sample_idx >= len(X_test):
                print(f"  ✗ Invalid sample index {sample_idx}")
                return None, None
            
            X_sample = X_test[sample_idx:sample_idx+1]
            X_bg = self._prepare_input(self.X_background)
            X_sample_prepared = self._prepare_input(X_sample)
            
            print(f"  Computing SHAP values for sample {sample_idx}...")
            
            # Create explainer
            explainer = shap.KernelExplainer(
                model=lambda x: self.model.predict(x, verbose=0),
                data=X_bg
            )
            
            # Get SHAP values for this sample
            shap_values = explainer.shap_values(X_sample_prepared)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Set feature names
            if self.feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_sample_prepared.shape[1])]
            else:
                feature_names = self.feature_names
            
            # Get top contributing features (top 5)
            shap_vals_sample = shap_values[0]  # Single sample's SHAP values
            abs_shap = np.abs(shap_vals_sample)
            top_indices = np.argsort(abs_shap)[-5:][::-1]  # Top 5 indices
            
            # Create explanation dict
            explanation_dict = {
                'predicted_proba': model_proba if model_proba is not None else 0.5,
                'true_label': 'Malware' if y_true == 1 else 'Benign' if y_true == 0 else 'Unknown',
                'top_features': []
            }
            
            for idx in top_indices:
                explanation_dict['top_features'].append({
                    'feature': feature_names[int(idx)] if int(idx) < len(feature_names) else f"Feature_{idx}",
                    'value': float(X_sample_prepared[0, int(idx)]),
                    'shap_value': float(shap_vals_sample[int(idx)]),
                    'direction': 'Increases Malware' if shap_vals_sample[int(idx)] > 0 else 'Increases Benign'
                })
            
            # Create bar plot for top features
            fig, ax = plt.subplots(figsize=(10, 6))
            
            top_shap_values = shap_vals_sample[top_indices]
            top_names = [feature_names[int(i)] if int(i) < len(feature_names) else f"Feature_{i}" 
                        for i in top_indices]
            
            # Color: red for positive (malware), blue for negative (benign)
            colors = ['red' if v > 0 else 'blue' for v in top_shap_values]
            
            ax.barh(range(len(top_shap_values)), top_shap_values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_shap_values)))
            ax.set_yticklabels(top_names)
            ax.set_xlabel('SHAP Value (contribution to prediction)')
            ax.set_ylabel('Features')
            ax.set_title(f'Local Explanation - Sample #{sample_idx}')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.invert_yaxis()
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='Pushes toward Malware'),
                Patch(facecolor='blue', alpha=0.7, label='Pushes toward Benign')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            print("  ✓ Local explanation generated successfully")
            
            return fig, explanation_dict
            
        except Exception as e:
            print(f"  ✗ Error in local explanation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


def explain_model(model, X_train: np.ndarray, X_test: np.ndarray, 
                  feature_names: Optional[List[str]] = None) -> SHAPExplainer:
    """
    Factory function to create and initialize SHAP explainer.
    
    Args:
        model: Trained CNN model
        X_train: Training data (for background)
        X_test: Test data (for explanation)
        feature_names: Feature names (optional)
        
    Returns:
        Initialized SHAPExplainer object
    """
    print("\n" + "="*70)
    print("INITIALIZING SHAP EXPLAINER FOR CNN MALWARE DETECTION")
    print("="*70)
    
    explainer = SHAPExplainer(model, X_train, feature_names)
    
    print("="*70 + "\n")
    return explainer


def display_global_explanation(explainer: SHAPExplainer, X_test: np.ndarray):
    """
    Display global SHAP explanation in Streamlit.
    
    Args:
        explainer: SHAPExplainer object
        X_test: Test data
    """
    st.subheader("🌍 Global Feature Importance (SHAP)")
    st.markdown("""
    This shows which features are most important for malware detection
    across all samples. Features at the top have the biggest influence
    on the model's decisions.
    """)
    
    with st.spinner("Computing SHAP values for feature importance..."):
        fig = explainer.explain_global(X_test, max_display=15)
        
        if fig:
            st.pyplot(fig)
            st.success("✓ Global explanation complete")
        else:
            st.error("✗ Could not generate global explanation")


def display_local_explanation(explainer: SHAPExplainer, X_test: np.ndarray, 
                             y_pred: np.ndarray, y_pred_proba: np.ndarray,
                             y_test: Optional[np.ndarray] = None):
    """
    Display local SHAP explanation for selected sample in Streamlit.
    
    Args:
        explainer: SHAPExplainer object
        X_test: Test data
        y_pred: Model predictions (class)
        y_pred_proba: Model probabilities
        y_test: True labels (optional)
    """
    st.subheader("🔍 Local Explanation (Individual Prediction)")
    st.markdown("""
    Select a sample to see why the model made its prediction.
    The plot shows which features contributed to the decision.
    """)
    
    # Sample selection
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_idx = st.slider("Select sample:", 0, len(X_test) - 1, 0)
    
    with col2:
        if st.button("Explain"):
            explain_button_pressed = True
        else:
            explain_button_pressed = False
    
    if explain_button_pressed or sample_idx >= 0:
        # Get prediction info
        pred_label = "Malware" if y_pred[sample_idx] == 1 else "Benign"
        pred_proba = y_pred_proba[sample_idx]
        true_label = None
        if y_test is not None:
            true_label = y_test[sample_idx]
        
        # Display prediction info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", pred_label)
        with col2:
            st.metric("Confidence", f"{pred_proba:.2%}")
        with col3:
            if true_label is not None:
                true_name = "Malware" if true_label == 1 else "Benign"
                st.metric("True Label", true_name)
        
        st.markdown("---")
        
        # Generate explanation
        with st.spinner("Computing SHAP explanation for this sample..."):
            fig, explanation_dict = explainer.explain_local(
                X_test, sample_idx, 
                model_proba=pred_proba,
                y_true=true_label
            )
            
            if fig and explanation_dict:
                # Display plot
                st.pyplot(fig)
                
                # Display top features
                st.subheader("Top 5 Contributing Features")
                
                for i, feature_info in enumerate(explanation_dict['top_features'], 1):
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col1:
                        st.write(f"**{i}. {feature_info['feature']}**")
                    with col2:
                        st.write(f"Value: {feature_info['value']:.4f}")
                    with col3:
                        direction = feature_info['direction']
                        if "Malware" in direction:
                            st.write(f"🔴 {direction}")
                        else:
                            st.write(f"🟢 {direction}")
                
                st.success("✓ Local explanation complete")
            else:
                st.error("✗ Could not generate local explanation")
