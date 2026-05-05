#!/usr/bin/env python3
"""Create synthetic malware dataset for testing the pipeline"""

import numpy as np
import pandas as pd

print("Generating synthetic malware dataset...")

np.random.seed(42)

# Parameters
n_samples = 2000
n_features = 100

# Generate features
X = np.random.randn(n_samples, n_features)

# Create label with realistic class imbalance (70% benign, 30% malware)
y = np.zeros(n_samples, dtype=int)
malware_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
y[malware_indices] = 1

# Add some signal: features 0-4 slightly predictive of malware
X[malware_indices, :5] += 2.0
X[~(y.astype(bool)), :5] -= 0.5

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
df['label'] = y

# Save
df.to_csv('malware_dataset.csv', index=False)

print(f"✅ Created synthetic dataset:")
print(f"   - {n_samples} samples")
print(f"   - {n_features} features")
print(f"   - Class distribution: {(y==0).sum()} benign, {(y==1).sum()} malware")
print(f"   - No data leakage (features weakly predictive)")
print(f"   - File: malware_dataset.csv")
