import pandas as pd

# Load extracted radiomics features
df = pd.read_csv('/home/ilants/radiomics6_project/up_scale/segmentation_masks/radiomics_features.csv')
meta = pd.read_csv("/home/ilants/radiomics6_project/up_scale/data_lung cancer/manifest-1608669183333/PET_CT_Metadata_with_Metastasis_Labels.csv")

# Merge with metastasis label
meta = meta[['PatientID', 'Metastasis_Label']].drop_duplicates()
df = df.merge(meta, on='PatientID')

# Keep only numeric radiomic features
X = df[[col for col in df.columns if col.startswith("original_")]]
y = df['Metastasis_Label']

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize features (required before PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (retain 95% of variance)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced to {X_pca.shape[1]} components")


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))



