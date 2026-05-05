"""
Data Preprocessing Module for RAM Forensics Malware Detection

KEY FEATURES:
1. Multi-file CSV merge support (handles mismatched columns)
2. Proper 70/15/15 train/val/test split BEFORE scaling (no data leakage)
3. Scaler fit ONLY on training data
4. Duplicate detection and removal (including near-duplicate check after split)
5. Dataset size validation & class distribution analysis
6. Data quality report generation for dashboard display
7. Proper error handling and logging
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import hashlib
from typing import Tuple, List, Optional, Dict


class DataPreprocessor:
    """Handles all data preprocessing operations with robust error handling."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler_train = MinMaxScaler()
        self.scaler_fitted = False
        
        self.label_column = None
        self.feature_names = None
        self.original_df = None
        self.merged_df = None
        self.class_distribution = None
        self.data_report = {}  # Data quality report for UI
        
    def merge_multiple_csv_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Merge multiple CSV files into a single dataset.
        
        Handles:
        - Missing columns (fills with NaN, then imputes)
        - Different column orders
        - Duplicate rows
        - Data type consistency
        
        Args:
            file_paths (List[str]): List of paths to CSV files
            
        Returns:
            pd.DataFrame: Merged dataset
            
        Raises:
            ValueError: If files cannot be merged or are empty
        """
        print("\n" + "="*70)
        print("MERGING MULTIPLE CSV FILES")
        print("="*70)
        
        if not file_paths or len(file_paths) == 0:
            raise ValueError("No files provided for merging")
        
        if len(file_paths) == 1:
            print(f"\n[OK] Single file provided, loading: {os.path.basename(file_paths[0])}")
            return pd.read_csv(file_paths[0])
        
        # Load all files
        dataframes = []
        all_columns = set()
        
        print(f"\nLoading {len(file_paths)} files...")
        for i, file_path in enumerate(file_paths, 1):
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                df = pd.read_csv(file_path)
                
                if df.empty:
                    print(f"  [WARN] File {i} is empty, skipping: {os.path.basename(file_path)}")
                    continue
                
                print(f"  [OK] File {i}: {len(df)} rows × {len(df.columns)} columns - {os.path.basename(file_path)}")
                dataframes.append(df)
                all_columns.update(df.columns)
                
            except Exception as e:
                print(f"  [ERR] Error loading file {i}: {str(e)}")
                raise
        
        if not dataframes:
            raise ValueError("No valid files to merge")
        
        # Align columns across all files
        print(f"\nAligning columns across files...")
        print(f"  Total unique columns: {len(all_columns)}")
        
        aligned_dfs = []
        for i, df in enumerate(dataframes, 1):
            # Add missing columns with NaN
            missing_cols = all_columns - set(df.columns)
            if missing_cols:
                print(f"  File {i}: Adding {len(missing_cols)} missing columns")
                for col in missing_cols:
                    df[col] = np.nan
            
            # Reorder to match
            df = df[[col for col in sorted(all_columns)]]
            aligned_dfs.append(df)
        
        # Concatenate all files
        print(f"\nConcatenating files...")
        merged = pd.concat(aligned_dfs, axis=0, ignore_index=True)
        print(f"  [OK] Merged: {len(merged)} total rows")
        
        # Remove complete duplicates
        initial_rows = len(merged)
        merged = merged.drop_duplicates()
        duplicates_dropped = initial_rows - len(merged)
        if duplicates_dropped > 0:
            print(f"  [WARN] Dropped {duplicates_dropped} duplicate rows")
        
        print("="*70 + "\n")
        return merged
    
    def load_dataset(self, file_path_or_list) -> pd.DataFrame:
        """
        Load single or multiple CSV files.
        
        Args:
            file_path_or_list: Single file path or list of file paths
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            # Handle multiple files
            if isinstance(file_path_or_list, list):
                df = self.merge_multiple_csv_files(file_path_or_list)
            else:
                # Single file
                if not os.path.exists(file_path_or_list):
                    raise FileNotFoundError(f"Dataset file not found: {file_path_or_list}")
                
                if not file_path_or_list.lower().endswith('.csv'):
                    raise ValueError("File must be in CSV format")
                
                df = pd.read_csv(file_path_or_list)
                print(f"\n[OK] Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            if len(df.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns (features + label)")
            
            # CRITICAL CHECK: Dataset size
            if len(df) < 1000:
                print(f"\n[WARN] WARNING: Dataset has only {len(df)} rows")
                print(f"           Minimum recommended: 1000 rows for reliable training")
            
            self.original_df = df.copy()
            return df
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")
    
    def detect_label_column(self, df: pd.DataFrame) -> str:
        """
        Automatically detect the label column.
        
        Args:
            df: The dataset
            
        Returns:
            str: Name of the label column
        """
        common_label_names = [
            'Class', 'class', 'Label', 'label', 'Target', 'target',
            'Malware', 'malware', 'Type', 'type', 'Category', 'category',
            'Result', 'result', 'Classification', 'classification'
        ]
        
        # Check for common names
        for col_name in common_label_names:
            if col_name in df.columns:
                self.label_column = col_name
                print(f"[OK] Label column detected: '{col_name}'")
                return col_name
        
        # Try by unique values (label columns have few unique values)
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if 0.01 < unique_ratio < 0.5:
                if df[col].dtype == 'object' or df[col].nunique() <= 10:
                    self.label_column = col
                    print(f"[OK] Label column detected (heuristic): '{col}'")
                    return col
        
        raise ValueError(f"Could not auto-detect label column. Columns: {df.columns.tolist()}")
    
    def convert_labels(self, df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """
        Convert labels to binary format (Benign=0, Malware=1).
        
        Args:
            df: The dataset
            label_column: Name of the label column
            
        Returns:
            pd.DataFrame: DataFrame with converted labels
        """
        df = df.copy()
        unique_values = df[label_column].unique()
        print(f"[OK] Found {len(unique_values)} unique label values: {list(unique_values)[:5]}")
        
        label_mapping = {}
        for val in unique_values:
            val_lower = str(val).lower().strip()
            
            if any(x in val_lower for x in ['benign', 'legitimate', 'clean', 'normal', '0']):
                label_mapping[val] = 0
            elif any(x in val_lower for x in ['malware', 'malicious', 'spam', 'trojan', 'bot', '1']):
                label_mapping[val] = 1
        
        if len(label_mapping) == 0:
            unique_vals = sorted(df[label_column].unique())
            for i, val in enumerate(unique_vals):
                label_mapping[val] = i % 2
        
        print(f"[OK] Label mapping: {label_mapping}")
        df[label_column] = df[label_column].map(label_mapping)
        
        if df[label_column].isnull().any():
            print("[WARN] Some labels couldn't be mapped, filling with mode")
            df[label_column] = df[label_column].fillna(df[label_column].mode()[0])
        
        class_dist = df[label_column].value_counts()
        print(f"[OK] Final class distribution:")
        print(f"    Benign (0): {class_dist.get(0, 0)} samples")
        print(f"    Malware (1): {class_dist.get(1, 0)} samples")
        
        self.class_distribution = class_dist
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """
        Handle missing values and duplicates.
        
        Args:
            df: The dataset
            label_column: Name of the label column
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        df = df.copy()
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"[WARN] Found {duplicates} duplicate rows - REMOVING")
            df = df.drop_duplicates()
        
        self.data_report['duplicates_removed'] = int(duplicates)
        
        # Check missing
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            print("[OK] No missing values")
            self.data_report['missing_values'] = 0
            return df.reset_index(drop=True)
        
        print(f"[WARN] Found {missing_count} missing values")
        self.data_report['missing_values'] = int(missing_count)
        
        # Drop rows with missing labels
        initial_rows = len(df)
        df = df.dropna(subset=[label_column])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"  • Dropped {dropped_rows} rows with missing labels")
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Drop remaining rows with any missing values
        df = df.dropna()
        
        return df.reset_index(drop=True)
    
    def extract_features(self, df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and labels (WITHOUT scaling yet).
        
        Args:
            df: The dataset
            label_column: Name of the label column
            
        Returns:
            tuple: (X unscaled, y labels)
        """
        X = df.drop(columns=[label_column]).copy()
        y = df[label_column].copy().reset_index(drop=True)
        
        # Keep only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols].reset_index(drop=True)
        
        print(f"[OK] Extracted {len(X.columns)} features, {len(X)} samples")
        
        self.feature_names = list(X.columns)
        self.data_report['num_features'] = len(X.columns)
        self.data_report['num_samples'] = len(X)
        return X, y
    
    def detect_feature_leakage(self, X: pd.DataFrame, y: pd.Series,
                                corr_threshold: float = 0.95) -> pd.DataFrame:
        """
        Detect and remove features that are suspiciously highly correlated
        with the label, which would cause perfect separation (data leakage).
        
        Args:
            X: Feature DataFrame (unscaled)
            y: Label Series
            corr_threshold: Absolute correlation threshold (default 0.95)
            
        Returns:
            pd.DataFrame: Cleaned X with leaked features removed
        """
        print("\n" + "="*70)
        print("FEATURE LEAKAGE DETECTION")
        print("="*70)
        
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        leaked = correlations[correlations > corr_threshold]
        
        self.data_report['feature_correlations_top10'] = [
            {'feature': str(feat), 'correlation': round(float(corr), 4)}
            for feat, corr in correlations.head(10).items()
        ]
        
        if len(leaked) > 0:
            leaked_names = list(leaked.index)
            print(f"  [WARN] FEATURE LEAKAGE FOUND! {len(leaked)} feature(s) with |corr| > {corr_threshold}:")
            for feat, corr in leaked.items():
                print(f"    - {feat}: corr = {corr:.4f}")
            
            X = X.drop(columns=leaked_names)
            self.feature_names = list(X.columns)
            self.data_report['leaked_features'] = [
                {'feature': str(f), 'correlation': round(float(c), 4)}
                for f, c in leaked.items()
            ]
            self.data_report['num_features_after_leakage_removal'] = len(X.columns)
            print(f"  [OK] Dropped {len(leaked_names)} leaked feature(s). Remaining: {len(X.columns)}")
        else:
            print(f"  [OK] No feature leakage detected (threshold: |corr| > {corr_threshold})")
            self.data_report['leaked_features'] = []
            self.data_report['num_features_after_leakage_removal'] = len(X.columns)
        
        # Also report features with moderate correlation (> 0.7) as warnings
        moderate = correlations[(correlations > 0.7) & (correlations <= corr_threshold)]
        if len(moderate) > 0:
            print(f"  [WARN] {len(moderate)} feature(s) with moderate correlation (0.7-{corr_threshold}):")
            for feat, corr in moderate.head(5).items():
                print(f"    - {feat}: corr = {corr:.4f}")
            self.data_report['moderate_corr_features'] = [
                {'feature': str(f), 'correlation': round(float(c), 4)}
                for f, c in moderate.items()
            ]
        else:
            self.data_report['moderate_corr_features'] = []
        
        print("="*70 + "\n")
        return X
    
    def _check_near_duplicates(self, X_train: np.ndarray, X_test: np.ndarray) -> int:
        """
        Check for near-duplicate samples between train and test sets.
        Uses row hashing (rounded to 4 decimal places) for efficiency.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            int: Number of near-duplicate samples found
        """
        def hash_row(row):
            rounded = np.round(row, 4)
            return hashlib.md5(rounded.tobytes()).hexdigest()
        
        train_hashes = set(hash_row(row) for row in X_train)
        duplicates = sum(1 for row in X_test if hash_row(row) in train_hashes)
        
        return duplicates
    
    def split_and_scale(self, X: pd.DataFrame, y: pd.Series, 
                       test_size: float = 0.15,
                       val_size: float = 0.15,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        CRITICAL FOR PREVENTING DATA LEAKAGE:
        1. Split FIRST (before scaling) — 70/15/15
        2. Fit scaler ONLY on training data
        3. Apply scaler to val and test data
        4. Check for near-duplicates between splits
        
        Args:
            X: Features (unscaled)
            y: Labels
            test_size: Test split proportion (default 0.15)
            val_size: Validation split proportion (default 0.15)
            random_state: Random seed
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
        """
        print("\n" + "="*70)
        print("TRAIN/VAL/TEST SPLIT & SCALING (Preventing Data Leakage)")
        print("="*70)
        
        # Convert to numpy
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        y_array = y_array.flatten()
        
        # Step 1: Split into train+val and test (stratified)
        combined_val_test_size = test_size + val_size  # 0.30
        print(f"\nStep 1: Stratified split — {(1-combined_val_test_size)*100:.0f}% train, "
              f"{val_size*100:.0f}% val, {test_size*100:.0f}% test")
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_array, y_array,
            test_size=test_size,
            random_state=random_state,
            stratify=y_array,
            shuffle=True
        )
        
        # Step 2: Split train_val into train and val
        val_ratio = val_size / (1 - test_size)  # e.g., 0.15 / 0.85 ≈ 0.176
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_train_val,
            shuffle=True
        )
        
        print(f"  [OK] Training set:   {X_train.shape[0]} samples ({X_train.shape[1]} features)")
        print(f"  [OK] Validation set: {X_val.shape[0]} samples")
        print(f"  [OK] Test set:       {X_test.shape[0]} samples")
        print(f"  [OK] Train — Benign: {(y_train==0).sum()}, Malware: {(y_train==1).sum()}")
        print(f"  [OK] Val   — Benign: {(y_val==0).sum()}, Malware: {(y_val==1).sum()}")
        print(f"  [OK] Test  — Benign: {(y_test==0).sum()}, Malware: {(y_test==1).sum()}")
        
        # Step 3: Check for near-duplicates between train and test
        print("\nStep 2: Checking for near-duplicate leakage...")
        train_test_dups = self._check_near_duplicates(X_train, X_test)
        train_val_dups = self._check_near_duplicates(X_train, X_val)
        
        if train_test_dups > 0:
            print(f"  [WARN] WARNING: {train_test_dups} near-duplicate samples found between train and test!")
        else:
            print(f"  [OK] No near-duplicates between train and test")
            
        if train_val_dups > 0:
            print(f"  [WARN] WARNING: {train_val_dups} near-duplicate samples found between train and val!")
        else:
            print(f"  [OK] No near-duplicates between train and val")
        
        self.data_report['train_test_near_duplicates'] = train_test_dups
        self.data_report['train_val_near_duplicates'] = train_val_dups
        
        # Step 4: Fit scaler ONLY on training data
        print("\nStep 3: Fitting scaler ONLY on training data...")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print(f"  [OK] Scaler trained on {X_train.shape[0]} training samples")
        
        # Step 5: Apply to val and test data (using training statistics)
        print("\nStep 4: Applying scaler to val and test data...")
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print(f"  [OK] Val and test sets scaled using training statistics (NO leakage)")
        
        self.scaler_train = scaler
        self.scaler_fitted = True
        
        # Store split info in report
        self.data_report['train_size'] = int(X_train.shape[0])
        self.data_report['val_size'] = int(X_val.shape[0])
        self.data_report['test_size'] = int(X_test.shape[0])
        self.data_report['train_class_dist'] = {
            'benign': int((y_train == 0).sum()),
            'malware': int((y_train == 1).sum())
        }
        self.data_report['val_class_dist'] = {
            'benign': int((y_val == 0).sum()),
            'malware': int((y_val == 1).sum())
        }
        self.data_report['test_class_dist'] = {
            'benign': int((y_test == 0).sum()),
            'malware': int((y_test == 1).sum())
        }
        
        print("="*70 + "\n")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def preprocess(self, file_path_or_list, 
                  test_size: float = 0.15,
                  val_size: float = 0.15,
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
        """
        Complete preprocessing pipeline with data leakage prevention.
        
        PIPELINE:
        1. Load and merge files
        2. Clean data (duplicates, missing values)
        3. Convert labels
        4. Extract features (NO scaling yet)
        5. SPLIT train/val/test (BEFORE scaling) — 70/15/15
        6. Fit scaler ONLY on training data
        7. Apply scaler to all sets
        8. Generate data quality report
        
        Args:
            file_path_or_list: Single file path or list of paths
            test_size: Test split proportion
            val_size: Validation split proportion
            random_state: Random seed
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, data_report)
        """
        print("\n" + "="*70)
        print("STARTING COMPLETE PREPROCESSING PIPELINE")
        print("="*70)
        
        self.data_report = {}  # Reset report
        
        try:
            # Step 1: Load dataset
            df = self.load_dataset(file_path_or_list)
            
            # Step 2: Detect and convert labels
            label_column = self.detect_label_column(df)
            df = self.convert_labels(df, label_column)
            
            # Step 3: Handle missing/duplicates
            df = self.handle_missing_values(df, label_column)
            print(f"[OK] After cleaning: {len(df)} rows")
            
            # Class imbalance check
            class_counts = df[label_column].value_counts()
            majority = class_counts.max()
            minority = class_counts.min()
            imbalance_ratio = majority / minority if minority > 0 else float('inf')
            self.data_report['class_imbalance_ratio'] = round(imbalance_ratio, 2)
            self.data_report['class_counts'] = {
                'benign': int(class_counts.get(0, 0)),
                'malware': int(class_counts.get(1, 0))
            }
            
            if imbalance_ratio > 3.0:
                print(f"\n[WARN] CLASS IMBALANCE DETECTED: ratio = {imbalance_ratio:.1f}:1")
                print(f"  Benign: {class_counts.get(0, 0)}, Malware: {class_counts.get(1, 0)}")
                print(f"  Class weights will be used during training\n")
            
            # Final dataset size check
            if len(df) < 1000:
                print(f"\n[WARN][WARN][WARN] WARNING: Dataset only has {len(df)} rows")
                print(f"    For reliable ML results, recommend at least 1000 rows")
                print(f"    Consider collecting more data [WARN][WARN][WARN]\n")
            
            # Step 4: Extract features
            X, y = self.extract_features(df, label_column)
            
            # Step 4b: Detect and remove feature leakage
            X = self.detect_feature_leakage(X, y, corr_threshold=0.95)
            
            # Step 5: Split and scale (70/15/15)
            X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = self.split_and_scale(
                X, y, test_size=test_size, val_size=val_size, random_state=random_state
            )
            
            print("="*70)
            print("[OK][OK][OK] PREPROCESSING COMPLETED SUCCESSFULLY [OK][OK][OK]")
            print("="*70 + "\n")
            
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, self.feature_names, self.data_report
            
        except Exception as e:
            print(f"\n[ERR] ERROR during preprocessing: {str(e)}")
            raise


def get_preprocessor() -> DataPreprocessor:
    """Factory function to get a preprocessor instance."""
    return DataPreprocessor()
