## 🧠 Segmentation & Radiomic Feature Extraction

This folder includes scripts and outputs related to segmenting mid-slices of PET/CT images and extracting radiomic features for radiomics-based metastasis classification.

---

## 🧰 Tools Used

- **MedSAM** – Foundation model for medical image segmentation (ViT-based)
- **PyRadiomics** – Feature extraction from image and mask
- **MongoDB Atlas** – Storing segmentation summaries and patient-level data

---

## 📁 Folder Contents

| File | Description |
|------|-------------|
| `lung_cancer_segmentation_ct.py` | Segmentation + feature extraction for CT scans (all patients) |
| `lung_cancer_segmentation_pet_ct.py` | Segmentation + feature extraction for both CT and PET (subset of 73 patients) |
| `ct_radiomics_features.csv` | Features from CT scans of all patients |
| `pet_ct_radiomics_features_CT.csv` | Features from CT scans of PET/CT patients |
| `pet_ct_radiomics_features_PET.csv` | Features from PET scans of PET/CT patients |

---

## 🧪 How It Works

Both scripts follow this general pipeline:

1. Load patient DICOM scans (PET/CT)
2. Identify the mid-slice of each scan
3. Run **MedSAM** to segment the image
4. Use **PyRadiomics** to extract features from the segmented area
5. Store features in:
   - CSV files
   - MongoDB (`patients` collection)

---

## 🧭 Which Script to Use?

| Script | Use Case |
|--------|----------|
| `lung_cancer_segmentation_ct.py` | Use this if you're processing **all CT scans** in the LUNG-PET-CT-DX dataset |
| `lung_cancer_segmentation_pet_ct.py` | Use this if you're processing only the **73 patients with both PET and CT** scans |

---

## ⚙️ How to Run

### 1. Install dependencies:

pip install -r requirements.txt

### 2. Add your MongoDB URI
Create a .env file in the project root with:

MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/

### 3. Place model weights
To use the MedSAM model for segmentation, download the pre-trained weights from the official GitHub repository:

📦 Checkpoint file: medsam_vit_b.pth
📁 Save location: ./Medsam/medsam_vit_b.pth

You can download it from:
🔗 https://github.com/bowang-lab/MedSAM#download-checkpoints

The vit_b version is used in this project.

### 4. Run the script
For CT-only:

python lung_cancer_segmentation_ct.py

For PET+CT subset:

python lung_cancer_segmentation_pet_ct.py

### 🗂 Output Structure
For CT-only script (segmentation_masks/):

segmentation_masks/
├── <patientID>_sliceXXX_image.png
├── <patientID>_sliceXXX_mask.png
├── <patientID>_sliceXXX_image_vis.png
├── <patientID>_sliceXXX_mask_vis.png

For PET/CT script (seg_files_ct_pet/):

seg_files_ct_pet/
├── <patientID>_CT_image.mha
├── <patientID>_CT_mask.png
├── <patientID>_PET_image.mha
├── <patientID>_PET_mask.png

# CSV outputs:

ct_radiomics_features.csv

pet_ct_radiomics_features_CT.csv

pet_ct_radiomics_features_PET.csv


## 📌 Notes
Mid-slice is selected for computational efficiency

PET and CT may require different preprocessing steps (windowing, normalization)

MongoDB stores patient-level segmentation summaries

Features include shape, texture, and intensity descriptors

Some CT images may have inconsistent slice dimensions – these are skipped

📚 Citation for MedSAM
Wang, B., Li, Y., Liu, Y., He, Z., You, C., Zhang, Y., ... & Chen, H. (2023).
MedSAM: Segment Anything in Medical Images
arXiv preprint arXiv:2306.00652
🔗 https://arxiv.org/abs/2306.00652







