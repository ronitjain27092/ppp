"""
CNN-Based Malware Detection Model - CLASS IMBALANCE FIX

This module fixes the critical class imbalance issue where the model
predicts only the majority class (Benign) and ignores the minority class (Malware).

ROOT CAUSE ANALYSIS:
====================
1. DATASET IMBALANCE: Benign samples >> Malware samples
   - Example: 9000 Benign vs 1000 Malware (90% vs 10%)
   - Model learns: "Always predict Benign" gets 90% accuracy
   - Problem: Malware never detected (Recall=0)

2. DEFAULT 0.5 THRESHOLD: Binary classification at 0.5
   - Model predicts: [0.15, 0.85] for sample
   - 0.85 > 0.5? No → Predict Benign
   - But 0.85 is actually HIGH malware probability!
   - Problem: Threshold too aggressive for imbalanced data

3. NO CLASS WEIGHTS: Model treats all misclassifications equally
   - Misclassifying 1 Malware = Misclassifying 1 Benign
   - But missing Malware is much worse!
   - Problem: No cost differentiation

SOLUTION:
=========
1. COMPUTE CLASS WEIGHTS
   - Calculate weight for each class based on frequency
   - Underrepresented class gets higher weight
   - Minimize loss: loss = class_weight[label] * categorical_cross_entropy

2. USE LOWER THRESHOLD
   - Instead of 0.5, use 0.3 or 0.2
   - More sensitive to minority class signals
   - Trades recall for precision (acceptable for malware)

3. PROPER CNN1D ARCHITECTURE
   - Use Conv1D for tabular/sequence data
   - Add Dropout (0.3-0.5) for regularization
   - Add BatchNormalization for stability
   - Keep model size manageable

4. VALIDATION DURING TRAINING
   - validation_split=0.2 monitors performance
   - EarlyStopping prevents overfitting
   - Watch for signs of bias (e.g., val recall=0)

5. DEBUG CHECKS
   - Print class distribution BEFORE training
   - Print computed class weights
   - Print predicted class counts AFTER inference
   - Alert if only one class is predicted
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns


class CNNMalwareDetector:
    """CNN model for malware detection with class imbalance handling."""
    
    def __init__(self, input_dim, model_name='cnn_malware_detector'):
        """
        Initialize the CNN model.
        
        Args:
            input_dim (int): Number of input features
            model_name (str): Name of the model
        """
        self.input_dim = input_dim
        self.model_name = model_name
        self.model = None
        self.history = None
        self.class_weights = None
        self.threshold = 0.3  # CRITICAL: Lower threshold for imbalanced data
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def compute_class_weights(self, y_train):
        """
        Compute class weights to handle imbalance.
        
        EXPLANATION:
        - Benign (class 0): Frequent → Lower weight
        - Malware (class 1): Rare → Higher weight
        
        Example:
        - If 9000 Benign, 1000 Malware:
          - Weight for Benign: 0.5
          - Weight for Malware: 4.5
        - This penalizes misclassifying Malware 9x more
        
        Args:
            y_train: Training labels
            
        Returns:
            dict: Class weights {0: weight_benign, 1: weight_malware}
        """
        print("\n" + "="*70)
        print("COMPUTING CLASS WEIGHTS FOR IMBALANCE MITIGATION")
        print("="*70)
        
        # Get unique classes
        classes = np.unique(y_train)
        
        # Compute weights
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        # Create dictionary
        class_weight_dict = {int(cls): float(wt) for cls, wt in zip(classes, weights)}
        
        # Print analysis
        print(f"\n✓ Class Distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = (count / len(y_train)) * 100
            weight = class_weight_dict[cls]
            label = "Benign" if cls == 0 else "Malware"
            print(f"  Class {cls} ({label:8}): {count:6} samples ({percentage:6.2f}%) → Weight: {weight:.4f}")
        
        # Explanation
        print(f"\n✓ Weight Interpretation:")
        print(f"  • Higher weight = More important for the model")
        print(f"  • Malware weight / Benign weight ratio: {class_weight_dict[1] / class_weight_dict[0]:.2f}x")
        print(f"  • Model will penalize Malware misclassification {class_weight_dict[1] / class_weight_dict[0]:.2f}x more")
        
        print("="*70 + "\n")
        
        self.class_weights = class_weight_dict
        return class_weight_dict
    
    def build_cnn_model(self):
        """
        Build CNN1D model optimized for tabular malware detection.
        
        ARCHITECTURE:
        - Input: Reshape features as 1D sequence
        - Conv1D layers: Extract local feature patterns
        - BatchNormalization: Stabilize training
        - Dropout: Reduce overfitting
        - GlobalAveragePooling: Aggregate features
        - Dense layers: Final classification
        
        KEY FEATURES FOR CLASS IMBALANCE:
        - Moderate model size (not too complex)
        - Dropout prevents memorizing minority class
        - BatchNorm helps training stability
        
        Returns:
            keras.Sequential: Compiled model
        """
        print("\n" + "="*70)
        print("BUILDING CNN1D MODEL (Optimized for Tabular Data)")
        print("="*70)
        
        model = Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),
            layers.Reshape((self.input_dim, 1)),  # Reshape to (features, channels)
            
            # Conv block 1: Extract initial patterns
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same',
                         name='conv1d_1'),
            layers.BatchNormalization(name='bn_1'),
            layers.Dropout(0.3, name='dropout_1'),  # 30% dropout
            
            # Conv block 2: Higher-level patterns
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same',
                         name='conv1d_2'),
            layers.BatchNormalization(name='bn_2'),
            layers.Dropout(0.3, name='dropout_2'),
            
            # Conv block 3: Fine-grained patterns
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same',
                         name='conv1d_3'),
            layers.BatchNormalization(name='bn_3'),
            layers.Dropout(0.3, name='dropout_3'),
            
            # Global pooling
            layers.GlobalAveragePooling1D(name='global_avg_pool'),
            
            # Dense layers
            layers.Dense(64, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001),
                        name='dense_1'),
            layers.Dropout(0.4, name='dropout_dense_1'),
            
            layers.Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001),
                        name='dense_2'),
            layers.Dropout(0.3, name='dropout_dense_2'),
            
            # Output layer: Binary classification
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        # Compile with appropriate loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # For binary classification
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),  # TP/(TP+FP)
                    keras.metrics.Recall(name='recall')]         # TP/(TP+FN)
        )
        
        print("\n✓ Model Architecture:")
        model.summary()
        
        print("\n✓ Key Features:")
        print("  • Conv1D layers: Extract local feature patterns")
        print("  • BatchNormalization: Stabilize training")
        print("  • Dropout (0.3): Reduce overfitting")
        print("  • Global pooling: Aggregate spatial information")
        print("  • L2 regularization: Prevent weight explosion")
        
        print("="*70 + "\n")
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=32, verbose=1):
        """
        Train the model with class weights and early stopping.
        
        CRITICAL PARAMETERS:
        - class_weight: Weights from compute_class_weights()
        - validation_split: 20% of training for validation
        - early_stopping: Stop if validation loss plateaus (patience=3)
        
        Args:
            X_train: Training features
            y_train: Training labels (0 or 1)
            epochs: Maximum epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        print("\n" + "="*70)
        print("TRAINING CNN MODEL WITH CLASS WEIGHTS")
        print("="*70)
        
        if self.class_weights is None:
            print("❌ ERROR: Call compute_class_weights() first!")
            raise ValueError("Class weights not computed. Call compute_class_weights()")
        
        # Ensure numpy arrays
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values.flatten() if hasattr(y_train, 'values') else np.array(y_train).flatten()
        
        print(f"\n✓ Training Configuration:")
        print(f"  • Training samples: {len(X_train_np)}")
        print(f"  • Features: {X_train_np.shape[1]}")
        print(f"  • Epochs: {epochs} (with EarlyStopping)")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Validation split: 20%")
        print(f"  • Class weights: {self.class_weights}")
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,  # Stop if val_loss doesn't improve for 3 epochs
            restore_best_weights=True,
            verbose=1
        )
        
        # Train with class weights
        print(f"\n✓ Training begins...")
        self.history = self.model.fit(
            X_train_np, y_train_np,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,  # 20% of training for validation
            class_weight=self.class_weights,  # CRITICAL: Apply class weights
            callbacks=[early_stop],
            verbose=verbose
        )
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETED")
        print(f"  • Actual epochs trained: {len(self.history.history['loss'])}")
        print(f"  • Final training loss: {self.history.history['loss'][-1]:.4f}")
        print(f"  • Final training accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"  • Final training recall: {self.history.history['recall'][-1]:.4f}")
        print("="*70 + "\n")
    
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluate model on test set with adjusted threshold.
        
        CRITICAL FIX:
        - Use custom threshold (NOT 0.5) for imbalanced data
        - Lower threshold (e.g., 0.3) makes model more sensitive to minority class
        - This increases recall for malware detection
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Custom threshold (default: self.threshold=0.3)
            
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*70)
        print("EVALUATING CNN MODEL (With Adjusted Threshold)")
        print("="*70)
        
        if threshold is not None:
            self.threshold = threshold
        
        # Ensure numpy arrays
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        y_test_np = y_test.values.flatten() if hasattr(y_test, 'values') else np.array(y_test).flatten()
        
        print(f"\n✓ Test Configuration:")
        print(f"  • Test samples: {len(X_test_np)}")
        print(f"  • Using threshold: {self.threshold}")
        print(f"  • Interpretation: P(malware) >= {self.threshold} → Predict Malware")
        
        # Get predictions
        self.y_pred_proba = self.model.predict(X_test_np, verbose=0)
        self.y_pred = (self.y_pred_proba >= self.threshold).astype(int).flatten()
        self.y_test = y_test_np
        
        # DEBUG: Check class distribution in predictions
        print(f"\n✓ Prediction Class Distribution:")
        unique_pred, counts_pred = np.unique(self.y_pred, return_counts=True)
        for cls, count in zip(unique_pred, counts_pred):
            percentage = (count / len(self.y_pred)) * 100
            label = "Benign" if cls == 0 else "Malware"
            print(f"  Predicted {cls} ({label:8}): {count:6} samples ({percentage:6.2f}%)")
        
        # ⚠️ ALERT if only one class predicted
        if len(unique_pred) == 1:
            print(f"\n⚠️  WARNING: Model predicts only class {unique_pred[0]}!")
            print(f"    • This indicates threshold/model bias")
            print(f"    • Consider lowering threshold further")
        
        # Calculate metrics
        print(f"\n✓ Computing Metrics...")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test_np, self.y_pred)
        precision = precision_score(y_test_np, self.y_pred, zero_division=0)
        recall = recall_score(y_test_np, self.y_pred, zero_division=0)
        f1 = f1_score(y_test_np, self.y_pred, zero_division=0)
        
        # ROC-AUC from probabilities
        try:
            roc_auc = roc_auc_score(y_test_np, self.y_pred_proba)
        except:
            roc_auc = 0.5
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        # Print metrics
        print(f"\n📊 TEST SET PERFORMANCE METRICS:")
        print(f"  • Accuracy:  {accuracy:.4f} - % correct predictions")
        print(f"  • Precision: {precision:.4f} - % predicted malware is truly malware")
        print(f"  • Recall:    {recall:.4f} - % of actual malware detected ⭐ KEY METRIC")
        print(f"  • F1-Score:  {f1:.4f} - Harmonic mean of precision & recall")
        print(f"  • ROC-AUC:   {roc_auc:.4f} - Overall discrimination ability")
        
        # Classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test_np, self.y_pred,
                                   target_names=['Benign', 'Malware'],
                                   zero_division=0))
        
        print("="*70 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix.
        
        INTERPRETATION:
        - TN (top-left): Correctly identified Benign
        - FP (top-right): Benign misclassified as Malware (false alarm)
        - FN (bottom-left): Malware missed (CRITICAL!)
        - TP (bottom-right): Correctly detected Malware
        """
        if self.y_test is None or self.y_pred is None:
            print("⚠ No predictions available")
            return None
        
        try:
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['Benign', 'Malware'],
                       yticklabels=['Benign', 'Malware'])
            plt.title(f'Confusion Matrix (Threshold: {self.threshold})', 
                     fontsize=12, fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Confusion matrix saved to {save_path}")
            
            return plt.gcf()
        
        except Exception as e:
            print(f"⚠ Error plotting confusion matrix: {str(e)}")
            return None
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        if self.history is None:
            print("⚠ No training history available")
            return None
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Accuracy
            axes[0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
            axes[0].plot(self.history.history['val_accuracy'], label='Val', linewidth=2)
            axes[0].set_title('Accuracy', fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Recall (KEY for malware detection)
            axes[1].plot(self.history.history['recall'], label='Train', linewidth=2, color='orange')
            axes[1].plot(self.history.history['val_recall'], label='Val', linewidth=2, color='red')
            axes[1].set_title('Recall (Malware Detection Rate)', fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Recall')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Loss
            axes[2].plot(self.history.history['loss'], label='Train', linewidth=2)
            axes[2].plot(self.history.history['val_loss'], label='Val', linewidth=2)
            axes[2].set_title('Loss', fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Loss (with class weights)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Training history saved to {save_path}")
            
            return fig
        
        except Exception as e:
            print(f"⚠ Error plotting training history: {str(e)}")
            return None
