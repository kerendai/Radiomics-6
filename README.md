# 🧬 Radiomics6 – Metastasis Prediction from CT and PET Imaging

Welcome to **Radiomics6**, a multi-stage radiomics pipeline for segmenting PET/CT images, extracting radiomic features, and training ML models to predict metastasis in cancer patients.

This project is part of the **Radiomics 6** initiative, leveraging open-source tools and public imaging datasets.

---

## 🧪 Project Overview

### 🎯 Goal:
Predict metastasis status using radiomic features extracted from CT and PET scans.

### 📊 Pipeline Steps:

1. **📥 Data Collection**  
   - Dataset: [LUNG-PET-CT-DX (TCIA)](https://www.cancerimagingarchive.net/collections/lung-pet-ct-dx)  
   - Includes whole-body CT and PET-CT scans + metadata

2. **🧠 Segmentation + Feature Extraction**  
   - Tools: `MedSAM`, `PyRadiomics`  
   - CT-only (all patients)  
   - PET+CT subset (73 patients)

3. **📈 Feature Analysis + Modeling**  
   - PCA for dimensionality reduction  
   - p-value feature selection  
   - ML Models: Logistic Regression, SVM, Random Forest, XGBoost, etc.

---

## 🧩 Branch Structure

| Branch | Description |
|--------|-------------|
| `data` | Dataset source and folder structure for LUNG-PET-CT-DX |
| `segmentation_features` | Scripts for segmentation (MedSAM) and radiomic feature extraction (PyRadiomics) |
| `modeling` | Correlation analysis, p-value computation, and training ML models |

---

## 🧰 Tools & Libraries

- Python, PyRadiomics, OpenCV, SimpleITK  
- MedSAM (segment-anything for medical imaging)  
- scikit-learn, imblearn (SMOTE), XGBoost  
- MongoDB Atlas (optional)

---

## ⚙️ How to Run (across branches)

1. Install dependencies:
pip install -r requirements.txt

2. Run segmentation (see Segmentation_and_feature_extraction branch)

Train models (see Classification_models_and_evaluation branch)

Results are saved as .csv/.xlsx files per model

### 📚 Citations
📁 Dataset
Clark K, Vendt B, Smith K, et al. (2013)
The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository
Journal of Digital Imaging. 26(6):1045–1057.
DOI: 10.1007/s10278-013-9622-7

🧠 MedSAM
Wang, B., Li, Y., Liu, Y., et al. (2023).
MedSAM: Segment Anything in Medical Images
arXiv:2306.00652
🔗 https://arxiv.org/abs/2306.00652

📬 Contact
Developed by Keren Dai and Lisa Cohen
For questions, please open an issue or contact directly.

