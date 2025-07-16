import os
import cv2
import pydicom
import numpy as np
import torch
import imageio
import csv
import SimpleITK as sitk
from radiomics import featureextractor
from segment_anything import sam_model_registry, SamPredictor
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import shutil


# === SETUP ===
root_dir = "./Lung-PET-CT-Dx"
output_root = "./sorted_dicoms"
os.makedirs(output_root, exist_ok=True)
checkpoint_path = "./Medsam/medsam_vit_b.pth"
mask_output_dir = "./seg_files_ct_pet"
os.makedirs(mask_output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# MedSAM
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

# PyRadiomics
extractor = featureextractor.RadiomicsFeatureExtractor()
csv_paths = {
    "CT": os.path.join(mask_output_dir, "radiomics_features_CT.csv"),
    "PET": os.path.join(mask_output_dir, "radiomics_features_PET.csv")
}
write_headers = {mod: not os.path.exists(csv_paths[mod]) for mod in csv_paths}

def load_image(path):
    return sitk.ReadImage(path, sitk.sitkInt16)

def run_medsam_segmentation(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    predictor.set_image(image)
    h, w, _ = image.shape
    box = np.array([w * 0.25, h * 0.25, w * 0.75, h * 0.75])
    masks, scores, logits = predictor.predict(box=box, multimask_output=False)
    return masks[0]

def get_series_by_modality(patient_root):
    ct_series = []
    pet_series = []

    for session in os.listdir(patient_root):
        session_path = os.path.join(patient_root, session)
        if not os.path.isdir(session_path):
            continue

        for scan in os.listdir(session_path):
            scan_path = os.path.join(session_path, scan)
            if not os.path.isdir(scan_path):
                continue

            name = scan.lower()
            if "pet" in name:
                pet_series.append(scan_path)
            elif "ct" in name and "b70" not in name:
                ct_series.append(scan_path)

    return ct_series, pet_series

'''for patient_folder in sorted(os.listdir(root_dir)):
    patient_path = os.path.join(root_dir, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    patient_id = patient_folder.replace("Lung_Dx-", "")
    print(f"\n📦 Copying DICOMs for patient: {patient_id}")

    ct_paths, pet_paths = get_series_by_modality(patient_path)

    for modality, paths in [("CT", ct_paths), ("PET", pet_paths)]:
        modality_outdir = os.path.join(output_root, patient_id, modality)
        os.makedirs(modality_outdir, exist_ok=True)

        file_counter = 0

        for series_path in paths:
            dicom_files = [f for f in os.listdir(series_path) if f.lower().endswith(".dcm")]
            for f in dicom_files:
                src = os.path.join(series_path, f)
                dst = os.path.join(modality_outdir, f"{patient_id}_{modality}_{file_counter:04d}.dcm")
                shutil.copyfile(src, dst)
                file_counter += 1

        print(f"✅ {modality}: Copied {file_counter} DICOM files to {modality_outdir}")'''

# === MAIN LOOP ===
for patient_id in sorted(os.listdir(output_root)):
    patient_dir = os.path.join(output_root, patient_id)
    if not os.path.isdir(patient_dir):
        continue

    print(f"\n--- Processing patient: {patient_id} ---")

    for modality in ["CT", "PET"]:
        modality_dir = os.path.join(patient_dir, modality)
        if not os.path.isdir(modality_dir):
            continue

        dicom_files = sorted([f for f in os.listdir(modality_dir) if f.lower().endswith(".dcm")])
        if not dicom_files:
            print(f"⚠ No DICOM files found for {patient_id} - {modality}")
            continue

        slices = []
        expected_shape = None

        for f in dicom_files:
            try:
                dcm = pydicom.dcmread(os.path.join(modality_dir, f), force=True)
                if hasattr(dcm, "pixel_array"):
                    arr = dcm.pixel_array
                    if expected_shape is None:
                        expected_shape = arr.shape
                    if arr.shape != expected_shape:
                        continue
                    slices.append(arr)
            except Exception as e:
                print(f"  ❌ Error reading {f}: {e}")

        if not slices:
            print(f"  ⚠ No valid slices for segmentation.")
            continue

        image_volume = np.stack(slices, axis=0)
        mid_slice_index = image_volume.shape[0] // 2
        mid_slice = image_volume[mid_slice_index]

        slice_uint8 = cv2.normalize(mid_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        try:
            segmentation_mask = run_medsam_segmentation(slice_uint8)
            print(f"  ✅ Segmentation done. Mask sum: {np.sum(segmentation_mask)}")
        except Exception as e:
            print(f"  ❌ MedSAM segmentation failed: {e}")
            continue

        if np.sum(segmentation_mask) == 0:
            print(f"  ⚠ Empty mask. Skipping.")
            continue

        # Convert RGB to grayscale if needed
        if mid_slice.ndim == 3 and mid_slice.shape[0] == 3:
            mid_slice = np.moveaxis(mid_slice, 0, -1)  # (H, W, 3)
            mid_slice = cv2.cvtColor(mid_slice, cv2.COLOR_RGB2GRAY)
        elif mid_slice.ndim == 3 and mid_slice.shape[2] == 3:
            mid_slice = cv2.cvtColor(mid_slice, cv2.COLOR_RGB2GRAY)

        # Create SimpleITK images (2D)
        image_sitk = sitk.GetImageFromArray(mid_slice.astype(np.uint16))
        mask_sitk = sitk.GetImageFromArray(segmentation_mask.astype(np.uint8))

        # Optional: save images
        image_path = os.path.join(mask_output_dir, f"{patient_id}_{modality}_image.mha")
        mask_path = os.path.join(mask_output_dir, f"{patient_id}_{modality}_mask.png")
        sitk.WriteImage(image_sitk, image_path)
        imageio.imwrite(mask_path, segmentation_mask.astype(np.uint8))

        print("Image size:", image_sitk.GetSize())
        print("Mask size:", mask_sitk.GetSize())
        print("Image dimension:", image_sitk.GetDimension())
        print("Mask dimension:", mask_sitk.GetDimension())

        # Feature extraction
        features = extractor.execute(image_sitk, mask_sitk)

        # Write to CSV
        with open(csv_paths[modality], mode="a", newline='') as f_csv:
            writer = csv.writer(f_csv)
            if write_headers[modality]:
                writer.writerow(["PatientID", "Modality"] + list(features.keys()))
                write_headers[modality] = False
            writer.writerow([patient_id, modality] + list(features.values()))

        # Save to MongoDB
        seg_summary = {
            "shape": segmentation_mask.shape,
            "sum": int(np.sum(segmentation_mask)),
            "center_slice_index": int(mid_slice_index),
            "mask_file": mask_path
        }
        patient_data = {
            "PatientID": patient_id,
            "Modality": modality,
            "Segmentation_Summary": seg_summary
        }
        patients_col.insert_one(patient_data)
        print(f"  📦 Stored features in MongoDB.")
