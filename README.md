# Welcome to Radiomics6!

This repository offers a complete pipeline for:

- Segmenting mid-slices from CT & PET/CT scans using **MedSAM**
- Extracting radiomic features using **PyRadiomics**
- Training classification models to predict **metastasis status**

This is part of the Radiomics6 research initiative focusing on AI-assisted imaging analysis.

---

## 🔖 Repository Contents

1. `lung_cancer_segmentation.py` – segmentation and feature extraction  
2. `train_models.py` – training and evaluation of models  
3. `radiomics_features.csv` – extracted features  
4. `model_predictions.xlsx` – predictions  
5. `model_results_summary.csv` – performance metrics  
6. `PET_CT_Metadata_with_Metastasis_Labels.csv` – patient-level labels  
7. `requirements.txt` – installation dependencies

## 📂 What the Code Does

### *lung_cancer_segmentation.py*
- Loads metadata and PET/CT DICOM scans
- Segments mid-slice with MedSAM
- Extracts radiomic features with PyRadiomics
- Saves masks and features to disk
- Inserts metadata and features into MongoDB

### *train_models.py*
- Loads radiomics features and labels
- Applies PCA for dimensionality reduction
- Trains Logistic Regression, SVM, and Random Forest
- Outputs model results and predictions

---

## ⚠️ IMPORTANT Notes
> **⚠️ Update File Paths**  
> Be sure to modify file paths according to your local data structure in both Python scripts.

## 🗃️ Dataset Source
The data used in this project is from the LUNG-PET-CT-Dx dataset, publicly available on The Cancer Imaging Archive (TCIA).

You can access the dataset here:
🔗 https://www.cancerimagingarchive.net/collections/lung-pet-ct-dx

### 📌 This dataset is in the public domain but may require account registration and download through the NBIA Data Retriever.

---

## 🛠 Requirements
Install dependencies with:

pip install -r requirements.txt

---

## 📁 Output Files
Segmentation results: segmentation_masks/

Radiomics features: radiomics_features.csv

Model results:

model_predictions.xlsx

model_results_summary.csv

---

## 🔮 Future Work
Here are some optional next steps for anyone continuing this project:

### 1. Organ Segmentation for Full-Body Analysis
Extend segmentation beyond tumors to include whole-body organs such as:
Liver
Kidneys
Spine
This enables more holistic analysis and cross-organ correlations using tools like TotalSegmentator or MedSAM.

### 2. Integrate Genomic Data
Combine imaging features with genomic mutations to support multi-omics models:
Common mutations: p53, BRCA1/2, MET
Enables correlation of phenotypic imaging traits with genetic profiles

### 3. Explore Multi-Modal Learning
Use multiple data types to enhance model robustness:
Combine PET, CT, metadata, and (optionally) genomic data
Apply advanced deep learning architectures that handle multiple modalities (e.g., transformers, late-fusion models)

---

## 📬 Contact
For questions or collaborations, please reach out via the repository’s issues page.

## 📝 License
This project is licensed under the MIT License — see the LICENSE file for details.

### Dataset Usage Notice
This project uses imaging data from the LUNG-PET-CT-DX dataset provided by The Cancer Imaging Archive (TCIA).

Please cite the dataset appropriately:

### Clark K, Vendt B, Smith K, et al. (2013)
The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository
Journal of Digital Imaging. 26(6):1045-1057.
DOI: 10.1007/s10278-013-9622-7

Use of this dataset is governed by the TCIA Data Usage Policy.
Be sure to comply with all licensing, citation, and ethical requirements outlined by TCIA.


