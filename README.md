# KNN Imputer in Machine Learning

## 🔹 What is KNN Imputer?
`KNN Imputer` is a technique to handle **missing values** in a dataset using the **K-Nearest Neighbors (KNN) algorithm**.  
It fills the missing values based on the similarity of the data points (neighbors).

- Each missing value is imputed by looking at the **k nearest neighbors** of that sample.  
- Neighbors are selected based on a **distance metric** (like Euclidean distance).  
- The imputed value is the **mean (or weighted mean)** of those neighbors.

---

## 🔹 When to use KNN Imputer?
- When dataset has **missing values (NaN)**.  
- When data has **similar patterns** that can be used for imputation.  
- Better than mean/median imputation because it **preserves data variation**.

---

## 🔹 Example in Python

```python
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Sample dataset with missing values
data = {
    'Age': [25, 27, np.nan, 35, 40, np.nan],
    'Salary': [50000, 54000, 58000, np.nan, 62000, 60000],
    'Experience': [1, 3, 4, 5, np.nan, 7]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Apply KNN Imputer
imputer = KNNImputer(n_neighbors=2)   # k = 2 nearest neighbors
df_imputed = imputer.fit_transform(df)

# Convert back to DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
print("\nImputed Data:\n", df_imputed)
