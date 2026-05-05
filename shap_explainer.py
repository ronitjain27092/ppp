"""
SHAP Explainer for Malware Detection - Fixed for 2D Data

Provides both global and local SHAP explanations for the DNN malware detector.

KEY FIXES:
- Uses 2D data only (no 3D reshaping issues)
- KernelExplainer (model-agnostic, robust)
- Wrapper function for model predictions
- Proper error handling

SHAP VALUES EXPLAINED:
- Positive value: Feature increases detection toward MALWARE
- Negative value: Feature decreases detection (suggests BENIGN)
- Magnitude: Importance of feature for this prediction
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP explainer for malware detection - uses 2D data only."""
    
    def __init__(self, model, feature_names=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained Keras model
            feature_names: List of feature names
        """
        self.model = model
        # Convert feature_names to list and ensure it's not causing array ambiguity
        if feature_names is not None:
            self.feature_names = np.asarray(feature_names)
        else:
            self.feature_names = None
        self.explainer = None
        self.background_data = None
        self._model_predict_fn = None
    
    def _create_predict_function(self):
        """
        Create a prediction function that works with SHAP.
        
        Handles any input shape and converts to model's expected format.
        
        Returns:
            callable: Function that takes 2D array and returns predictions
        """
        def predict_fn(X):
            """Prediction function for SHAP - assumes 2D input."""
            X = np.asarray(X)
            
            # Ensure 2D shape
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Make prediction
            try:
                pred = self.model.predict(X, verbose=0)
                
                # Handle different output shapes
                if isinstance(pred, list):
                    pred = pred[-1]
                
                # Ensure 1D output (flatten if needed)
                if len(pred.shape) > 1:
                    pred = pred.flatten()
                
                return pred
            
            except Exception as e:
                raise RuntimeError(f"Model prediction failed: {str(e)}")
        
        return predict_fn
    
    def init_with_background_data(self, X_background):
        """
        Initialize explainer with background data.
        
        Args:
            X_background: Background samples (2D array: n_samples x n_features)
        """
        X_bg = self._prepare_data(X_background)
        
        # Validate 2D shape
        if len(X_bg.shape) != 2:
            raise ValueError(f"Background data must be 2D (samples, features), got shape {X_bg.shape}")
        
        # Ensure float32 type for SHAP
        X_bg = np.asarray(X_bg, dtype=np.float32)
        
        # Limit background to 30 samples for efficiency and to reduce SHAP computation
        if len(X_bg) > 30:
            indices = np.random.choice(len(X_bg), 30, replace=False)
            X_bg = X_bg[indices]
        
        self.background_data = X_bg
        
        # Create prediction function
        self._model_predict_fn = self._create_predict_function()
        
        # Create KernelExplainer (model-agnostic, robust)
        try:
            self.explainer = shap.KernelExplainer(
                model=self._model_predict_fn,
                data=self.background_data,
                link='logit'  # For probability outputs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SHAP explainer: {str(e)}")
    
    def _prepare_data(self, X, validate_nan=True):
        """
        Convert input to 2D numpy array with NaN validation.
        
        Args:
            X: Input data (DataFrame, list, or array)
            validate_nan: Check for NaN values and raise error if found
            
        Returns:
            np.ndarray: 2D array (n_samples, n_features)
            
        Raises:
            ValueError: If NaN values found or shape is invalid
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.asarray(X)
        
        # Ensure 2D shape
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        elif len(X.shape) > 2:
            raise ValueError(f"Data must be 2D, got shape {X.shape}")
        
        # Convert to float32 for consistency
        X = X.astype(np.float32)
        
        # Check for NaN values
        if validate_nan:
            nan_mask = np.isnan(X)
            if np.any(nan_mask):
                n_nans = np.sum(nan_mask)
                nan_positions = np.where(nan_mask)
                raise ValueError(
                    f"Data contains {n_nans} NaN value(s). "
                    f"Found at positions: {list(zip(nan_positions[0][:5], nan_positions[1][:5]))}. "
                    f"Please replace NaN values (try filling with 0, mean, or median)"
                )
        
        # Check for Inf values
        inf_mask = np.isinf(X)
        if np.any(inf_mask):
            raise ValueError(
                f"Data contains {np.sum(inf_mask)} infinite value(s). "
                f"Please replace Inf values with valid numbers (0-1 range)"
            )
        
        return X
    
    def _validate_and_clean_data(self, X, fill_nan_strategy='mean'):
        """
        Validate data and optionally clean NaN/Inf values.
        
        Args:
            X: Input data
            fill_nan_strategy: 'mean', 'median', 'zero', or None (raise error)
            
        Returns:
            np.ndarray: Cleaned data without NaN/Inf
        """
        X = self._prepare_data(X, validate_nan=False)
        
        # Check for NaN
        if np.any(np.isnan(X)):
            if fill_nan_strategy is None:
                raise ValueError("Data contains NaN values. Please clean data first.")
            
            # Fill NaN values
            if fill_nan_strategy == 'mean':
                col_means = np.nanmean(X, axis=0)
                col_means[np.isnan(col_means)] = 0  # If entire column is NaN
                for i in range(X.shape[1]):
                    X[np.isnan(X[:, i]), i] = col_means[i]
            
            elif fill_nan_strategy == 'median':
                col_medians = np.nanmedian(X, axis=0)
                col_medians[np.isnan(col_medians)] = 0
                for i in range(X.shape[1]):
                    X[np.isnan(X[:, i]), i] = col_medians[i]
            
            elif fill_nan_strategy == 'zero':
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Check for Inf
        if np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        return X
    
    def explain_instance(self, X_instance, num_samples=50, handle_nan=True):
        """
        Get SHAP explanation for a single instance (local explanation).
        
        OPTIMIZED for single predictions:
        - num_samples: Default 50 (faster) vs 100 (more accurate)
        - Computation time: ~30-60 seconds
        - Automatically handles NaN values if requested
        
        Args:
            X_instance: Single sample (1D or 2D array)
            num_samples: Number of samples for KernelExplainer (50=fast, 100=accurate)
            handle_nan: If True, automatically replace NaN with 0, else raise error
            
        Returns:
            dict: SHAP values, prediction, instance data, feature_names
            
        Raises:
            ValueError: If data contains NaN and handle_nan=False
            RuntimeError: If SHAP computation fails
            
        Example:
            >>> shap_exp = explainer.explain_instance(X_test[0:1], num_samples=50)
            >>> print(f"Prediction: {shap_exp['prediction']:.2%}")
            >>> print(f"Top feature: {shap_exp['feature_names'][0]}")
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call init_with_background_data first.")
        
        try:
            # Prepare data with NaN validation
            if handle_nan:
                # Automatically clean NaN values
                X = self._validate_and_clean_data(X_instance, fill_nan_strategy='zero')
            else:
                # Strict validation - raise error on NaN
                X = self._prepare_data(X_instance, validate_nan=True)
            
            # Ensure 2D shape
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            if len(X) > 1:
                X = X[0:1]  # Use only first sample
            
            # Validate model prediction
            try:
                prediction = float(self._model_predict_fn(X)[0])
                
                # Check if prediction is valid
                if np.isnan(prediction):
                    raise RuntimeError(
                        "Model returned NaN prediction. "
                        "This may indicate invalid input features or model issues. "
                        "Ensure all feature values are normalized (0.0-1.0) and not NaN/Inf."
                    )
            except Exception as pred_err:
                raise RuntimeError(f"Model prediction failed: {str(pred_err)}")
            
            # Compute SHAP values for this instance
            try:
                shap_values = self.explainer.shap_values(X, nsamples=num_samples)
            except Exception as shap_err:
                # Catch specific NaN errors and provide helpful message
                error_msg = str(shap_err).lower()
                if 'nan' in error_msg:
                    raise RuntimeError(
                        "SHAP computation encountered NaN values. "
                        "This may be due to: (1) Feature values outside 0-1 range, "
                        "(2) Model instability, (3) Insufficient training samples. "
                        f"Original error: {str(shap_err)}"
                    )
                else:
                    raise RuntimeError(f"SHAP computation failed: {str(shap_err)}")
            
            # Handle list output (binary classification returns list)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1] if len(shap_values) > 0 else shap_values[0]
            
            # Extract first sample's SHAP values
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # Validate SHAP values
            if np.any(np.isnan(shap_vals)):
                raise RuntimeError(
                    "SHAP values contain NaN. This indicates model or data issues. "
                    "Try: (1) Normalizing features to 0-1, (2) Retraining model, "
                    "(3) Using different background samples"
                )
            
            return {
                'shap_values': shap_vals,
                'prediction': prediction,
                'instance': X[0],
                'feature_names': self.feature_names,
                'num_samples': num_samples  # For reference
            }
        
        except (ValueError, RuntimeError) as e:
            # Re-raise our custom errors
            raise
        except Exception as e:
            raise RuntimeError(f"SHAP computation failed for instance: {str(e)}")
    
    def explain_instance_fast(self, X_instance, num_samples=25):
        """
        Fast local SHAP explanation (quicker computation).
        
        Uses fewer samples for faster computation (~15-30 seconds).
        Good for interactive demos and quick checks.
        
        Args:
            X_instance: Single sample (1D or 2D array)
            num_samples: Number of samples for KernelExplainer (default 25=very fast)
            
        Returns:
            dict: SHAP values, prediction, instance data, feature_names
        """
        return self.explain_instance(X_instance, num_samples=num_samples)
    
    def explain_batch(self, X_batch, max_samples=10, num_samples=100):
        """
        Get SHAP explanations for multiple instances (global explanation).
        
        Args:
            X_batch: Batch of samples (2D array)
            max_samples: Max samples to explain (for speed)
            num_samples: Samples for KernelExplainer (more = slower)
            
        Returns:
            dict: shap_values, X_batch, feature_names
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call init_with_background_data first.")
        
        X = self._prepare_data(X_batch)
        
        # Limit samples for speed (KernelExplainer is slow)
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X, nsamples=num_samples)
        
        # Handle list output
        if isinstance(shap_values, list):
            shap_values = shap_values[-1] if len(shap_values) > 0 else shap_values[0]
        
        return {
            'shap_values': shap_values,
            'X_batch': X,
            'feature_names': self.feature_names
        }
    
    def get_feature_importance(self, shap_batch_result, top_n=15):
        """
        Get top N features by absolute SHAP value.
        
        Args:
            shap_batch_result: Result from explain_batch()
            top_n: Number of top features
            
        Returns:
            pd.DataFrame: Features ranked by importance
        """
        shap_vals = shap_batch_result['shap_values']
        feature_names = shap_batch_result['feature_names']
        
        # Mean absolute SHAP = feature importance
        importance = np.abs(shap_vals).mean(axis=0)
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return df.head(top_n)
    
    def plot_summary(self, shap_batch_result, plot_type='bar', max_display=15):
        """
        Create SHAP summary bar plot.
        
        Args:
            shap_batch_result: Result from explain_batch()
            plot_type: 'bar' (recommended)
            max_display: Max features to show
            
        Returns:
            matplotlib.figure.Figure
        """
        shap_vals = shap_batch_result['shap_values']
        feature_names = shap_batch_result['feature_names']
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        try:
            # Mean absolute SHAP values = feature importance
            importance = np.abs(shap_vals).mean(axis=0)
            
            # Top features
            top_idx = np.argsort(importance)[-max_display:][::-1]
            top_importance = importance[top_idx]
            top_names = feature_names[top_idx]
            
            # Create bar plot
            y_pos = np.arange(len(top_idx))
            plt.barh(y_pos, top_importance, color='steelblue')
            plt.yticks(y_pos, top_names)
            plt.xlabel('Mean |SHAP value| (Feature Importance)', fontsize=12)
            plt.title('SHAP Global Feature Importance', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
        
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot Error: {str(e)}', ha='center', va='center',
                    transform=fig.transFigure)
        
        return fig
    
    def plot_waterfall(self, shap_instance_result):
        """
        Create waterfall plot for single prediction.
        
        Args:
            shap_instance_result: Result from explain_instance()
            
        Returns:
            matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=(12, 8))
        
        try:
            shap_vals = shap_instance_result['shap_values']
            feature_names = shap_instance_result['feature_names']
            
            # Get top features for waterfall
            top_idx = np.argsort(np.abs(shap_vals))[-10:][::-1]  # Top 10
            top_vals = shap_vals[top_idx]
            top_names = feature_names[top_idx]
            
            # Calculate cumulative sum
            cumsum = np.cumsum(top_vals)
            
            # Create waterfall plot
            colors = ['red' if v > 0 else 'blue' for v in top_vals]
            
            fig.clear()
            ax = fig.add_subplot(111)
            
            y_pos = np.arange(len(top_idx))
            ax.barh(y_pos, top_vals, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names)
            ax.set_xlabel('SHAP value (Feature Contribution)', fontsize=12)
            ax.set_title('SHAP Local Explanation - Top Contributing Features', 
                        fontsize=14, fontweight='bold')
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
        
        except Exception as e:
            plt.text(0.5, 0.5, f'Plot Error: {str(e)}', ha='center', va='center',
                    transform=fig.transFigure)
        
        return fig
    
    def get_top_contributing_features(self, shap_instance_result, top_n=5):
        """
        Get top N features contributing to prediction.
        
        Args:
            shap_instance_result: Result from explain_instance()
            top_n: Number of features
            
        Returns:
            pd.DataFrame: Top contributing features with impact
        """
        shap_vals = shap_instance_result['shap_values']
        feature_names = shap_instance_result['feature_names']
        
        # Top by absolute value
        top_indices = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
        
        df = pd.DataFrame({
            'Feature': feature_names[top_indices],
            'Impact': shap_vals[top_indices],
            'Direction': ['🔴 Malware' if shap_vals[i] > 0 else '🟢 Benign' 
                         for i in top_indices],
            'Magnitude': np.abs(shap_vals[top_indices])
        })
        
        return df.sort_values('Magnitude', ascending=False)

