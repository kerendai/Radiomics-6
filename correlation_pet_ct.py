
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# Load CT and PET radiomics features
ct_path = "./pet_ct__radiomics_features_CT.csv"
pet_path = "./pet_ct_radiomics_features_PET.csv"

ct = pd.read_csv(ct_path)
pet = pd.read_csv(pet_path)

# Match patients by ID
common_ids = set(ct["PatientID"]).intersection(set(pet["PatientID"]))
ct_matched = ct[ct["PatientID"].isin(common_ids)].sort_values("PatientID").reset_index(drop=True)
pet_matched = pet[pet["PatientID"].isin(common_ids)].sort_values("PatientID").reset_index(drop=True)

# Select only numeric radiomic features (excluding metadata and IDs)
ct_features = ct_matched.drop(columns=["PatientID", "Modality"]).select_dtypes(include='number')
pet_features = pet_matched.drop(columns=["PatientID", "Modality"]).select_dtypes(include='number')

# Correlation between matched CT and PET features
correlation = ct_features.corrwith(pet_features)

# Plot histogram of correlation values
plt.figure(figsize=(10, 4))
sns.histplot(correlation, bins=30, kde=True)
plt.title("Correlation Distribution Between Matched CT and PET Features")
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("correlation_distribution.png")
plt.close()

# Print correlation summary
print("\nSummary statistics of CT vs PET feature correlations:")
print(correlation.describe())

# Show top correlated feature pairs
print("\nTop 10 most positively correlated features:")
print(correlation.sort_values(ascending=False).head(10))

print("\nTop 10 most negatively correlated features:")
print(correlation.sort_values().head(10))

# PCA analysis for CT
scaler_ct = StandardScaler()
X_ct_scaled = scaler_ct.fit_transform(ct_features)

pca_ct = PCA(n_components=0.95, random_state=42)
X_ct_pca = pca_ct.fit_transform(X_ct_scaled)

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_ct.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("CT PCA - Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("ct_pca_variance.png")
plt.close()

# PCA analysis for PET
scaler_pet = StandardScaler()
X_pet_scaled = scaler_pet.fit_transform(pet_features)

pca_pet = PCA(n_components=0.95, random_state=42)
X_pet_pca = pca_pet.fit_transform(X_pet_scaled)

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_pet.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PET PCA - Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("pet_pca_variance.png")
plt.close()

print(f"\nCT PCA: selected {X_ct_pca.shape[1]} components to explain 95% variance.")
print(f"PET PCA: selected {X_pet_pca.shape[1]} components to explain 95% variance.")
