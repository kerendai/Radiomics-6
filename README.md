# 🧪 Modeling & Evaluation – CT and PET/CT

This folder contains scripts and outputs for radiomics-based metastasis prediction using machine learning models trained on CT and PET features.

---

## 🎯 Goals

- Analyze correlations between CT and PET radiomics features
- Reduce dimensionality with PCA
- Select informative features using p-values
- Train multiple classification models (Logistic Regression, SVM, Random Forest, etc.)
- Evaluate model performance using Accuracy, F1, and Log Loss

---

## 📁 Folder Contents

| File | Description |
|------|-------------|
| `correlation_pet_ct.py` | Correlation and PCA analysis between PET and CT features |
| `ct_train_models.py` | Trains models using CT-only radiomics features |
| `train_models_pet_ct.py` | Trains models using both PET and CT features with SMOTE |
| `PET_CT_Metadata_with_Metastasis_Labels.csv` | Patient labels (metastasis status) |
| `pet_ct_patient_correlations.csv` | Output of PET-CT feature correlation analysis |
| `p_values_PET_CT.xlsx` | Top significant features based on p-value |
| `model_predictions_ct.xlsx` | Per-patient predictions for each CT-only model |
| `model_results_summary_ct.csv` | Accuracy, F1-score, and Log Loss for CT-only models |
| `prt_ct_model_evaluation_results.csv` | Combined results from PET and CT models |

---

## 🤖 Models Used

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- K-Nearest Neighbors  
- Naive Bayes  
- Gradient Boosting  
- XGBoost (if installed)

All models were trained using PCA-reduced features explaining 95% of the variance.

---

## ⚙️ How to Run

### 1. Install dependencies:
```bash
pip install -r requirements.txt

׳׳׳

### 2. Run correlation analysis:

python correlation_pet_ct.py

### 3. Train CT-only models:

python ct_train_models.py

### 4. Train PET+CT models and perform evaluation:

python train_models_pet_ct.py

## 📊 Output Highlights
model_predictions_ct.xlsx
Per-patient model predictions and probabilities for CT-only models

model_results_summary_ct.csv
Summary table with Accuracy, F1-score, and Log Loss for CT models

prt_ct_model_evaluation_results.csv
Model performance metrics for both CT and PET (with SMOTE balancing)

p_values_PET_CT.xlsx
Top features significantly associated with metastasis (p-value < 0.05)

pet_ct_patient_correlations.csv
Pearson correlations between PET and CT feature sets

## 📌 Notes
PCA is applied separately for CT and PET to reduce feature dimensions

SMOTE is used in the PET/CT pipeline to balance class distribution

PET and CT pipelines are kept separate for comparison

XGBoost is optional – install it if needed




