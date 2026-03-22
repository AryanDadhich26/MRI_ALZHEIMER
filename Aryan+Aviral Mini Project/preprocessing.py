import os
import SimpleITK as sitk
import antspynet
import ants
import nibabel as nib
import torchio as tio
import numpy as np
from antspynet.utilities import brain_extraction
from nibabel.orientations import axcodes2ornt, inv_ornt_aff, apply_orientation

# Define root directories
root_folder = "CN"  # Main folder containing all subfolders
output_folder = "CN_preprocessed_slices"  # Output directory for processed NIfTI files

# Create main output directory and subfolders
os.makedirs(output_folder, exist_ok=True)
axial_folder = os.path.join(output_folder, "axial")
coronal_folder = os.path.join(output_folder, "coronal")
sagittal_folder = os.path.join(output_folder, "sagittal")

os.makedirs(axial_folder, exist_ok=True)
os.makedirs(coronal_folder, exist_ok=True)
os.makedirs(sagittal_folder, exist_ok=True)


def find_deepest_dicom_folders(root_folder):
    """Find all the deepest folders containing DICOM files."""
    dicom_folders = []

    for root, dirs, files in os.walk(root_folder):
        if any(f.lower().endswith(".dcm") for f in files):  
            dicom_folders.append(root)

    return dicom_folders

def convert_dicom_to_sitk_image(dicom_folder):
    """Convert the first DICOM series in a folder to a SimpleITK Image."""
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)

    if not series_IDs:
        print(f"⚠ No DICOM series found in: {dicom_folder}")
        return None

    if len(series_IDs) > 1:
        print(f"⚠ Multiple series found. Using the first one: {series_IDs[0]}")

    series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
    reader.SetFileNames(series_file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    return image

def save_subject_as_nifti(subject, save_path='CN_Preprocessed_slices', original_nifti_path=None):
    """
    Save a TorchIO Subject's MRI image as a NIfTI file.

    Args:
        subject (tio.Subject): The TorchIO subject with a ScalarImage.
        save_path (str): Path where the NIfTI file will be saved.
        original_nifti_path (str): Optional path to original NIfTI for header copying.
    """
    # Extract subject name from its attributes
    subject_name = getattr(subject, "name", "Unknown_Subject")  # Use 'Unknown_Subject' if no name found

    # Construct the save path with subject name
    save_path = os.path.join(save_path, f"{subject_name}_preprocessed.nii.gz")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    # Get image tensor and affine
    img_tensor = subject.mri.data  # (1, H, W, D)
    affine = subject.mri.affine    # (4, 4)

    # Remove channel dimension and convert to numpy
    img_np = img_tensor.squeeze().numpy()  # (H, W, D)

    # Optional: copy header from original NIfTI if provided
    header = None
    if original_nifti_path and os.path.exists(original_nifti_path):
        original_img = nib.load(original_nifti_path)
        header = original_img.header.copy()

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(img_np, affine, header=header)
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)  # Create folder if missing

    # Save to disk
    nib.save(nifti_img, save_path)
    print(f"✅ Saved preprocessed NIfTI to: {subject+save_path}")

def preprocess(dicom_folder):
    """Preprocess and register an MRI NIfTI file."""
    raw_img_sitk = convert_dicom_to_sitk_image(dicom_folder)
    if raw_img_sitk is None:
        return None
    # Get the current pixel type
    pixel_type = raw_img_sitk.GetPixelID()

    print(f"📌 Processing: {dicom_folder} | Pixel Type: {pixel_type}")

    # # If already 32-bit float, return directly
    # if pixel_type == sitk.sitkFloat32:
    #     print("✅ Image is already 32-bit float, skipping conversion.")
        
    if raw_img_sitk.GetPixelID() == sitk.sitkUInt64:
        
        # Convert to NumPy array
        raw_img_np = sitk.GetArrayFromImage(raw_img_sitk)

    # Ensure the data type is float32
        raw_img_np = raw_img_np.astype(np.float32)  

    # Convert back to SimpleITK image
        raw_img_sitk = sitk.GetImageFromArray(raw_img_np)

        print("✅ Image successfully converted to 32-bit float!")

    else:
        raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)
    # raw_img_sitk = convert_dicom_to_sitk_image(path)
    # Load DICOM series and convert to NIfTI
    # Read the NIfTI file using SimpleITK
    raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)
    raw_img_sitk = sitk.DICOMOrient(raw_img_sitk, 'IAL')

    # Denoise using SimpleITK
    denoised_img_sitk = sitk.CurvatureFlow(raw_img_sitk, timeStep=0.123, numberOfIterations=5)

    # Rescale Intensity & Thresholding
    transformed = sitk.RescaleIntensity(denoised_img_sitk, 0, 255)
    transformed = sitk.LiThreshold(transformed, 0, 1)
    head_mask = transformed

    # Bias Correction
    shrinkFactor = 4
    inputImage = sitk.Shrink(denoised_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)

    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(denoised_img_sitk)
    log_bias_field = sitk.Cast(log_bias_field, denoised_img_sitk.GetPixelID())
    corrected_image_full_resolution = denoised_img_sitk / sitk.Exp(log_bias_field)
    # Save Processed Image
    output_filename = "registered_check.nii"
    sitk.WriteImage(corrected_image_full_resolution, output_filename)
    # Convert to ANTsPy Image
    raw_img_ants = ants.image_read(output_filename, reorient='LAS')
    # Brain Extraction
    prob_brain_mask = brain_extraction(raw_img_ants, modality="t1", verbose=True)
    brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.4)
    masked = ants.mask_image(raw_img_ants, brain_mask)
    # Load MNI152 Template for Registration
    template_path = ants.get_ants_data("mni")
    template = ants.image_read(template_path)

    # Register Image to MNI Template
    registration = ants.registration(
        fixed=template,
        moving=masked,
        type_of_transform="SyN",
        verbose=True
    )

    # Apply Transformation
    registered_image = ants.apply_transforms(
        fixed=template,
        moving=masked,
        transformlist=registration["fwdtransforms"]
    )

    output_path = "registered_check.nii" # Use .nii for uncompressed or .nii.gz for compressed
    ants.image_write(registered_image, output_path)
    #############################################################
    img_obj = nib.load(output_path)
    ras_ornt = axcodes2ornt("RAS")
    current_ornt = nib.orientations.io_orientation(img_obj.affine)
    transform = nib.orientations.ornt_transform(current_ornt, ras_ornt)

    # Apply reorientation
    reoriented_data = apply_orientation(img_obj.get_fdata(), transform)
    new_affine = img_obj.affine @ inv_ornt_aff(transform, img_obj.shape)

    # Save the reoriented NIfTI file
    img_obj= nib.Nifti1Image(reoriented_data, new_affine)
    # Get data and affine
    img_data = img_obj.get_fdata()
    img_affine = img_obj.affine

    # Ensure correct shape for TorchIO: (C, H, W, D)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

    # Create TorchIO subject with proper affine
    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=img_data, affine=img_affine)
    )
    # Apply preprocessing
    preprocessing = tio.Compose([
        tio.Resample((1,1,1)),  # Ensure consistent voxel size
        tio.ZNormalization(),
        tio.CropOrPad((224,224,224),padding_mode='constant'),
        tio.RescaleIntensity((-1,1)),
        tio.RandomFlip(axes=(0,)),
        # tio.RandomAffine(scales=(0.9,1.1), degrees=15)
    ])

    subject = preprocessing(subject)

    save_subject_as_nifti(subject, save_path=output_path, original_nifti_path=output_path)
    return output_path


# preprocess(r"CN\002_S_0413\MPRAGE\2015-06-09_07_01_52.0\I569632")

import nibabel as nib
import numpy as np
import os

def save_slices(nifti_path, output_name, num_slices=15):
    """Extract and save 15 slices centered around the middle slice from Axial, Coronal, and Sagittal views."""
    
    nifti_image = nib.load(nifti_path)
    image_array = nifti_image.get_fdata()

    def get_center_slices(arr, axis, num_slices):
        """Extract exactly 'num_slices' centered around the middle slice."""
        total_slices = arr.shape[axis]
        middle = total_slices // 2  # Find the middle slice
        half_range = num_slices // 2
        
        start = max(middle - half_range, 0)  # Ensure not below 0
        end = min(middle + half_range + 1, total_slices)  # Ensure within range

        indices = np.arange(start, end)  # Consecutive slices
        return indices

    axial_indices = get_center_slices(image_array, axis=2, num_slices=num_slices)
    coronal_indices = get_center_slices(image_array, axis=1, num_slices=num_slices)
    sagittal_indices = get_center_slices(image_array, axis=0, num_slices=num_slices)

    # Function to save slices
    def save_slice(slice_array, folder, output_name, view, index):
        """Save a 2D slice as a NIfTI file."""
        slice_array = np.squeeze(slice_array)  # Ensure it's 2D
        slice_path = os.path.join(folder, f"{output_name}_{view}_slice_{index+1}.nii.gz")
        nib.save(nib.Nifti1Image(slice_array, np.eye(4)), slice_path)
        print(f"📂 Saved {view} slice {index+1}: {slice_path}")

    # Save 15 slices for each view
    for i, idx in enumerate(axial_indices):
        save_slice(image_array[:, :, idx], axial_folder, output_name, "axial", i)

    for i, idx in enumerate(coronal_indices):
        save_slice(image_array[:, idx, :], coronal_folder, output_name, "coronal", i)

    for i, idx in enumerate(sagittal_indices):
        save_slice(image_array[idx, :, :], sagittal_folder, output_name, "sagittal", i)

    print("✅ 15 slices centered around the middle saved successfully!")




def main(root_folder):
    """Find and process all DICOM folders."""
    dicom_folders = find_deepest_dicom_folders(root_folder)

    if not dicom_folders:
        print("⚠ No valid DICOM folders found.")
        return

    for dicom_folder in dicom_folders:
        print(f"🔄 Processing: {dicom_folder}")
        preprocessed_path = preprocess(dicom_folder)
        
        if preprocessed_path:
            output_name = os.path.basename(dicom_folder)
            save_slices(preprocessed_path, output_name)
        else:
            print(f"❌ Skipping {dicom_folder} due to processing error.")


# Run the pipeline
main(root_folder) 