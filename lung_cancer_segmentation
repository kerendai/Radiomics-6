"""
Script to segment mid-slices of PET/CT images using MedSAM, extract radiomic features,
and store results in MongoDB and local CSV/PNG formats.
"""
import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
import imageio
import csv
import SimpleITK as sitk
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from segment_anything import sam_model_registry, SamPredictor
from radiomics import featureextractor
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# === SETUP ===
metadata_path = './PET_CT_Metadata_with_Metastasis_Labels.csv'  
root_dir = "./Lung-PET-CT-Dx"
checkpoint_path = "./Medsam/medsam_vit_b.pth"
mask_output_dir = "./segmentation_masks"
os.makedirs(mask_output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load metadata
df = pd.read_csv(metadata_path)
# 'PatientID' is already a column
df['PatientID'] = df['PatientID'].astype(str)  # Ensure it's a string, just in case
# Fix paths without duplicating the top folder
df['Fixed Path'] = df['File Location'].str.lstrip('.\\').str.replace('\\', '/', regex=False)
df['Fixed Path'] = df['Fixed Path'].str.replace('Lung-PET-CT-Dx/', '', regex=False)  # <- new line
df['Full Path'] = df['Fixed Path'].apply(lambda x: os.path.join(root_dir, x))

# Connect to MongoDB Atlas
MONGODB_URI=mongodb+srv://your-username:your-password@your-cluster.mongodb.net/?retryWrites=true&w=majority
# Load environment variables from .env
load_dotenv()

# Retrieve the MongoDB URI from environment variable
uri = os.getenv("MONGODB_URI")

# Connect to MongoDB
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["radiomics6"]
patients_col = db["patients"]
patients_col.delete_many({})

# Load MedSAM
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

# PyRadiomics setup
extractor = featureextractor.RadiomicsFeatureExtractor()
csv_path = os.path.join(mask_output_dir, "radiomics_features.csv")
write_header = not os.path.exists(csv_path)

# Convert PNG to SimpleITK
def load_image(path):
    return sitk.ReadImage(path, sitk.sitkInt16)

# Run MedSAM
def run_medsam_segmentation(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    predictor.set_image(image)
    h, w, _ = image.shape
    box = np.array([w * 0.25, h * 0.25, w * 0.75, h * 0.75])  # Correct box format
    masks, scores, logits = predictor.predict(box=box, multimask_output=False)
    return masks[0]



# Main loop
with open(csv_path, mode="a", newline='') as f_csv:
    writer = csv.writer(f_csv)
    

    for idx, row in df.iterrows():
        patient_id = row['PatientID']
        series_path = row['Full Path']

        if not os.path.exists(series_path):
            print(f"[{patient_id}] Series path not found: {series_path}")
            continue

        dicom_files = sorted([f for f in os.listdir(series_path) if f.endswith('.dcm')])
        if not dicom_files:
            print(f"[{patient_id}] No DICOM files in {series_path}")
            continue

        slices = []

        expected_shape = None  # To store the shape of the first valid slice

        for f_dcm in dicom_files:
            try:
                dcm = pydicom.dcmread(os.path.join(series_path, f_dcm), force=True)
                if hasattr(dcm, "pixel_array"):
                    arr = dcm.pixel_array

                    if expected_shape is None:
                        expected_shape = arr.shape  # Set from first valid slice

                    if arr.shape != expected_shape:
                        print(f"[{patient_id}] Inconsistent slice shape in {f_dcm}: {arr.shape}, expected {expected_shape}")
                        continue

                    slices.append(arr)
                else:
                    print(f"[{patient_id}] File {f_dcm} has no pixel data.")
            except Exception as e:
                print(f"[{patient_id}] Error reading {f_dcm}: {str(e)}")

        if not slices:
            print(f"[{patient_id}] Failed to load slices.")
            continue

        image_volume = np.stack(slices, axis=0)
        mid_slice_index = image_volume.shape[0] // 2
        mid_slice = image_volume[mid_slice_index]

        # Rescale mid_slice to uint8 before segmentation
        slice_uint8 = cv2.normalize(mid_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        try:
            segmentation_mask = run_medsam_segmentation(slice_uint8)
        except Exception as e:
            print(f"[{patient_id}] MedSAM segmentation failed: {e}")
            continue  # Skip to next patient if MedSAM fails

        # Save mid-slice and mask
        image_filename = f"{patient_id}_slice{mid_slice_index}_image.png"
        mask_filename = f"{patient_id}_slice{mid_slice_index}_mask.png"
        image_path = os.path.join(mask_output_dir, image_filename)
        mask_path = os.path.join(mask_output_dir, mask_filename)
        # Save original image for radiomics
        imageio.imwrite(image_path, mid_slice.astype(np.uint16))

        # Save visual version of the image
        imageio.imwrite(image_path.replace(".png", "_vis.png"),
                        cv2.normalize(mid_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

        # Save mask (label = 1)
        imageio.imwrite(mask_path, (segmentation_mask * 1).astype(np.uint8))

        # Save visual version of the mask
        imageio.imwrite(mask_path.replace(".png", "_vis.png"), (segmentation_mask * 255).astype(np.uint8))


        # Extract radiomics
        image_sitk = load_image(image_path)
        mask_sitk = load_image(mask_path)
        features = extractor.execute(image_sitk, mask_sitk)

        # Write CSV
        if write_header:
            writer.writerow(["PatientID"] + list(features.keys()))
            write_header = False
        writer.writerow([patient_id] + list(features.values()))

        # Store in MongoDB
        seg_summary = {
            "shape": segmentation_mask.shape,
            "sum": int(np.sum(segmentation_mask)),
            "center_slice_index": int(mid_slice_index),
            "mask_file": mask_filename
        }

        patient_data = {
            "PatientID": patient_id,
            "Age": row.get("Age"),
            "Sex": row.get("Sex"),
            "M-Stage": row.get("M-Stage"),
            "Metastasis_Label": int(row.get("Metastasis_Label", -1)),
            "Series_Description": row.get("Series Description"),
            "Study_Date": row.get("Study Date"),
            "Series_Path": series_path,
            "Segmentation_Summary": seg_summary
        }

        patients_col.insert_one(patient_data)
        print(f"[{patient_id}] ✅ Segment, save, extract, store — DONE.")
