Welcome to Radiomics6!

In this repository, you will find the necessary data structure, scripts, and code needed to use PyRadiomics and execute our CT and PET/CT analysis pipeline on the Lung-PET-CT-DX dataset from TCIA. This project focuses on whole-body PET/CT analysis, segmentation, and radiomic feature extraction, followed by exploratory and predictive modeling.

How to run the pipeline:
Open the project in Python (or Jupyter Notebook). You can run the segmentation and feature extraction modules directly or integrate them into your own workflow.

📦 Required libraries:
- numpy
- pandas
- scikit-learn
- pyradiomics
- SimpleITK
- pymongo
- matplotlib

Import commands are already included in the scripts and notebooks.

---

### Stages for Execution:

#### PET/CT Images – Data Acquisition → Segmentation:

The original dataset is downloaded from [TCIA – Lung-PET-CT-DX](https://www.cancerimagingarchive.net/) in DICOM format. Each subject includes CT and PET volumes and annotations.

To segment organs or tumor areas, we use:

- **TotalSegmentator**: for automatic whole-body segmentation.
- **MedSAM**: for fine-tuned segmentation on medical data.

Segmentation outputs are saved as mask files aligned with the original CT/PET images.

---

#### PyRadiomics – Feature Extraction:

To extract features:

1. Convert the images and masks (if needed) to NIfTI format.
2. Prepare a CSV file listing:
   - Image path
   - Mask path
   - Patient ID or label

Run PyRadiomics using:

```bash
pyradiomics path/to/images_and_masks.csv -o output_features.csv -f csv
