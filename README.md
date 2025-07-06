Welcome to Radiomics6!

This repository provides a full pipeline for segmenting mid-slices from CT & PET/CT scans using MedSAM, extracting radiomic features using PyRadiomics, and training classification models to predict metastasis status. It is part of the Radiomics6 research project focused on enhancing medical imaging analysis through AI and radiomics.

🧩 Repository Structure

📁 lung_cancer_segmentation.py        # Performs segmentation, radiomics extraction, MongoDB storage
📁 train_models.py                    # Trains classifiers on extracted features
📄 PET_CT_Metadata_with_Metastasis_Labels.csv
📄 radiomics_features.csv
📄 model_results_summary.csv
📄 model_predictions.xlsx
📄 README.md

📂 What the Code Does

*lung_cancer_segmentation.py*

* Loads metadata and DICOM PET/CT scans

* Segments the mid-slice using MedSAM

* Extracts radiomic features using PyRadiomics

* Saves the segmented masks and features to disk

* Stores metadata and summary in MongoDB

*train_models.py*

* Loads the radiomics CSV and labels

* Applies PCA for dimensionality reduction

* Trains Logistic Regression, SVM, and Random Forest

* Saves prediction results and model performance metrics to disk

⚠️ IMPORTANT — Update File Paths
The file paths provided in the code are example paths that you should replace based on where your data and models are stored locally.
Make sure all referenced folders and files exist on your system.
If you're using a different folder structure, update the paths accordingly in both lung_cancer_segmentation.py and train_models.py.

🗃️ Dataset Source
The data used in this project is from the LUNG-PET-CT-Dx dataset, publicly available on The Cancer Imaging Archive (TCIA).

You can access the dataset here:
🔗 https://www.cancerimagingarchive.net/collections/lung-pet-ct-dx

! This dataset is in the public domain but may require account registration and download through the NBIA Data Retriever.

🛠 Requirements
To install the required packages, run:

pip install -r requirements.txt

✅ Output Files

Once the scripts are executed:

Segmentation results are saved in segmentation_masks/

Radiomics features are stored in radiomics_features.csv

Model performance is saved in:

model_predictions.xlsx

model_results_summary.csv

🔮 Future Work
Here are some optional next steps for anyone continuing this project:

1. Organ Segmentation for Full-Body Analysis
Extend segmentation beyond tumors to include whole-body organs such as:
Liver
Kidneys
Spine
This enables more holistic analysis and cross-organ correlations using tools like TotalSegmentator or MedSAM.

2. Integrate Genomic Data
Combine imaging features with genomic mutations to support multi-omics models:
Common mutations: p53, BRCA1/2, MET
Enables correlation of phenotypic imaging traits with genetic profiles

3. Explore Multi-Modal Learning
Use multiple data types to enhance model robustness:
Combine PET, CT, metadata, and (optionally) genomic data
Apply advanced deep learning architectures that handle multiple modalities (e.g., transformers, late-fusion models)

📬 Contact
For questions or collaborations, please reach out via the repository’s issues page.
