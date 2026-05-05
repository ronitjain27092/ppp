"""
Explainable AI Module using SHAP
- Generate SHAP explanations
- Feature importance visualization
- Summary plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


class SHAPExplainer:
    """Generate SHAP explanations for model predictions."""
    
    def __init__(self, model, feature_names):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Keras model
            feature_names (list): List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.X_sample = None
    
    def create_explainer(self, X_background, sample_size=100):
        """
        Create SHAP explainer using background data.
        
        Args:
            X_background: Background data for SHAP calculation
            sample_size (int): Size of background sample for efficiency
        """
        print("\n" + "="*60)
        print("CREATING SHAP EXPLAINER")
        print("="*60)
        
        # Use a sample of data for efficiency
        if len(X_background) > sample_size:
            indices = np.random.choice(len(X_background), sample_size, replace=False)
            X_sample = X_background.iloc[indices] if isinstance(X_background, pd.DataFrame) else X_background[indices]
        else:
            X_sample = X_background
        
        self.X_sample = X_sample
        
        # Create explainer - using Kernel SHAP for flexibility
        print("  • Creating Kernel SHAP explainer...")
        self.explainer = shap.KernelExplainer(
            model=self.model,
            data=shap.sample(X_sample, min(50, len(X_sample)))
        )
        
        print("✓ SHAP Explainer created successfully")
        print("="*60 + "\n")
    
    def explain_prediction(self, X_instance):
        """
        Generate SHAP explanation for single prediction.
        
        Args:
            X_instance: Single instance to explain (as numpy array or dataframe row)
            
        Returns:
            dict: Explanation dictionary with SHAP values and base value
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Ensure input is numpy array and 2D
        if hasattr(X_instance, 'values'):
            X_instance = X_instance.values
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        
        explanation = {
            'shap_values': shap_values[0] if isinstance(shap_values, list) else shap_values[0],
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value,
            'instance': X_instance[0]
        }
        
        return explanation
    
    def get_feature_importance(self, X_data, num_samples=None):
        """
        Calculate global feature importance using SHAP.
        
        Args:
            X_data: Data to explain (background set or test set)
            num_samples (int): Number of samples to explain (for efficiency)
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        print("\n" + "="*60)
        print("CALCULATING FEATURE IMPORTANCE")
        print("="*60)
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Limit samples for efficiency
        if num_samples and len(X_data) > num_samples:
            indices = np.random.choice(len(X_data), num_samples, replace=False)
            X_explain = X_data.iloc[indices] if isinstance(X_data, pd.DataFrame) else X_data[indices]
            print(f"  • Using {num_samples} samples for explanation (total: {len(X_data)})")
        else:
            X_explain = X_data
            print(f"  • Using all {len(X_data)} samples for explanation")
        
        # Calculate SHAP values
        print("  • Computing SHAP values...")
        shap_values = self.explainer.shap_values(X_explain)
        
        # Mean absolute SHAP values = feature importance
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        if len(shap_values.shape) == 3:
            shap_values = shap_values[1]  # For binary classification, take malware class
        
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n✓ Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']:30s} {row['Importance']:.4f}")
        
        print("="*60 + "\n")
        
        return importance_df
    
    def plot_summary(self, X_data, num_samples=50, save_path=None):
        """
        Create SHAP summary plot.
        
        Args:
            X_data: Data to explain
            num_samples (int): Number of samples to use
            save_path (str): Path to save the plot
        """
        print("Creating SHAP summary plot...")
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Limit for efficiency
        if len(X_data) > num_samples:
            indices = np.random.choice(len(X_data), num_samples, replace=False)
            X_explain = X_data.iloc[indices] if isinstance(X_data, pd.DataFrame) else X_data[indices]
        else:
            X_explain = X_data
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_explain)
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        shap.summary_plot(shap_values, X_explain, feature_names=self.feature_names,
                         show=False, plot_type="bar")
        
        plt.title('Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Summary plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_dependence(self, feature_name, X_data, num_samples=100, save_path=None):
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            feature_name (str): Feature to analyze
            X_data: Data to explain
            num_samples (int): Number of samples to use
            save_path (str): Path to save the plot
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not in feature list")
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        feature_idx = self.feature_names.index(feature_name)
        
        # Limit for efficiency
        if len(X_data) > num_samples:
            indices = np.random.choice(len(X_data), num_samples, replace=False)
            X_explain = X_data.iloc[indices] if isinstance(X_data, pd.DataFrame) else X_data[indices]
        else:
            X_explain = X_data
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_explain)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create dependence plot
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values, X_explain,
                            feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dependence plot saved to {save_path}")
        
        return plt.gcf()
    
    def explain_single_instance(self, instance, feature_names_input=None, save_path=None):
        """
        Create detailed explanation for single instance.
        
        Args:
            instance: Single sample to explain
            feature_names_input (list): Optional feature names for display
            save_path (str): Path to save the plot
            
        Returns:
            dict: Explanation with SHAP values and feature contributions
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Ensure numpy array and 2D shape
        if hasattr(instance, 'values'):
            instance = instance.values
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Get SHAP explanation
        shap_values = self.explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            shap_values_class = shap_values[1][0]  # Malware class
        else:
            shap_values_class = shap_values[0]
        
        # Create explanation
        explanation = {
            'features': self.feature_names,
            'values': instance[0],
            'shap_values': shap_values_class,
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
        }
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_values_class))
        
        colors = ['red' if x > 0 else 'blue' for x in shap_values_class[sorted_idx]]
        
        plt.barh(range(len(sorted_idx)), shap_values_class[sorted_idx], color=colors)
        plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
        plt.xlabel('SHAP Value (impact on prediction)')
        plt.title('Feature Contribution to Prediction\n(Red=increases malware score, Blue=decreases)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Instance explanation saved to {save_path}")
        
        return explanation, plt.gcf()


def create_explainer(model, feature_names):
    """Factory function to create explainer."""
    return SHAPExplainer(model, feature_names)
