"""
This script loads radiomics features and metastasis labels,
applies PCA, trains multiple classifiers (Logistic Regression, SVM, Random Forest),
and evaluates them. Results are saved as Excel and CSV files.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os

# Load data
df = pd.read_csv('path/to/radiomics_features.csv')
meta = pd.read_csv('path/to/PET_CT_Metadata_with_Metastasis_Labels.csv')
meta = meta[['PatientID', 'Metastasis_Label']].drop_duplicates()
df = df.merge(meta, on='PatientID')

# Extract features and labels
X = df[[col for col in df.columns if col.startswith("original_")]]
y = df['Metastasis_Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: keep enough components to retain 95% of variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {X_pca.shape[1]} components")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# Retrieve corresponding PatientIDs for test set
patient_ids_test = df.loc[y_test.index, 'PatientID'].reset_index(drop=True)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Directories and results storage
save_path = './results'
results = []
all_preds = []

# Train, evaluate, and store results for each model
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba) if y_proba is not None else None

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Log Loss: {ll if ll is None else f'{ll:.4f}'}")
    print(classification_report(y_test, y_pred))

    # Save metrics
    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1 Score': f1,
        'Log Loss': ll if ll is not None else 'N/A'
    })

    # Save predictions with PatientID
    pred_df = pd.DataFrame({
        'Model': name,
        'PatientID': patient_ids_test,
        'True Label': y_test.values,
        'Predicted Label': y_pred
    })
    if y_proba is not None:
        pred_df['Probability_0'] = y_proba[:, 0]
        pred_df['Probability_1'] = y_proba[:, 1]

    all_preds.append(pred_df)

# Save predictions to Excel (one sheet per model)
excel_path = os.path.join(save_path, 'model_predictions.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for pred_df in all_preds:
        model_name = pred_df['Model'].iloc[0].replace(" ", "_")
        pred_df.to_excel(writer, sheet_name=model_name[:31], index=False)

# Save model results summary
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_path, 'model_results_summary.csv'), index=False)
