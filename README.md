# MRI Alzheimer's Disease Classification

Deep learning based Alzheimer's Disease diagnosis using neuroimaging data. Final year mini project by Aryan Dadhich and Aviral Saini, LNMIIT, under the guidance of Dr. Rahul Sharma.

---

## Goal

Alzheimer's Disease progresses through stages that are hard to tell apart visually, especially in the earlier phases. This project builds a full pipeline that takes raw brain MRI scans and automatically classifies them into four diagnostic stages:

- **CN**: Cognitively Normal
- **EMCI**: Early Mild Cognitive Impairment
- **MCI**: Mild Cognitive Impairment
- **AD**: Alzheimer's Disease

The aim is to explore whether deep learning can support faster, more consistent staging than manual radiological reading alone.

---

## Dataset

- **Source:** ADNI (Alzheimer's Disease Neuroimaging Initiative), a real-world clinical neuroimaging dataset used widely in Alzheimer's research.
- **Format:** Raw structural MRI scans in DICOM format.
- **Final curated dataset:** 18,030 MRI slices, distributed evenly across the four diagnostic classes.
- **Slice extraction:** 15 centered slices taken from each of the axial, coronal, and sagittal planes per scan, giving a multi-view representation of each brain.

---

## Preprocessing Pipeline

Raw DICOM scans are noisy, inconsistently oriented, and vary across scanners and patients. The preprocessing pipeline (`preprocessing.py`) cleans and standardizes every scan before it reaches a model.

1. **DICOM to image conversion**: Convert DICOM series to a SimpleITK image, cast to float32.
2. **Denoising**: Apply SimpleITK's CurvatureFlow filter to reduce scan noise.
3. **Skull stripping**: Use ANTsPyNet's brain extraction model to isolate brain tissue from skull and surrounding structures.
4. **N4 bias field correction**: Correct scanner-induced intensity inhomogeneity across the volume.
5. **Spatial registration**: Register each scan to the MNI-152 anatomical template using SyN registration via ANTsPy, so every brain is spatially aligned.
6. **Reorientation**: Reorient volumes to RAS coordinate space for consistency.
7. **Normalization (TorchIO)**: Resample to 1mm isotropic voxels, apply Z-normalization, crop or pad to 224x224x224, rescale intensity, and apply flip augmentation.
8. **Slice extraction**: Pull 15 centered slices from each of the three anatomical planes per scan.

---

## Model Architectures

Two families of models were built and compared.

**Custom CNNs (trained from scratch)**
- CNN-5, 5-layer convolutional network
- CNN-10, 10-layer convolutional network
- CNN-15, 15-layer convolutional network

**Pretrained backbones (transfer learning, ImageNet weights)**
- ResNet50
- ResNet101
- EfficientNet-B3

---

## Training Details

- **Class weighting** to handle any class imbalance.
- **Early stopping** and **learning rate scheduling** to prevent overfitting.
- **Deterministic seeding** for reproducible results.
- **Evaluation suite** including confusion matrices, ROC curves, precision/recall/F1, and full classification reports, not just raw accuracy.

Example training command:

```bash
python main.py --family pretrained --model resnet101 --epochs 200
```

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Custom CNNs (5/10/15 layer) | 31% to 59% | Shallow, trained from scratch |
| ResNet50 | 50% | Pretrained backbone |
| EfficientNet-B3 | 52% | Validation AUC of 0.80 |
| **ResNet101** | **91%** | Macro F1-score of 0.91, per-class AUC 0.98 to 0.99 |

ResNet101 was the clear best performer, with a confusion matrix showing minimal misclassification between adjacent disease stages.

---

## Tech Stack

**Languages & core frameworks:** Python, TensorFlow, Keras, PyTorch

**Medical imaging & preprocessing:** TorchIO, ANTsPy, ANTsPyNet, SimpleITK, NiBabel

**Evaluation & tuning:** Scikit-learn, Keras Tuner

---

## Key Learnings

- Preprocessing quality has an outsized effect on final model performance in medical imaging, a bad skull-stripping or registration step quietly damages everything downstream.
- Transfer learning clearly outperforms training from scratch on smaller, harder-to-collect medical datasets like this one.
- Evaluating with confusion matrices and per-class metrics matters more than a single accuracy number, especially in a healthcare context where misclassifying disease stages carries real risk.

---

## Repository Structure

```
Aryan+Aviral Mini Project/
├── preprocessing.py                     # DICOM to standardized MRI slice pipeline
├── Mini_Project_Report.pdf              # Full written report with architecture and results
├── Deep-Learning-for-Alzheimers-Disease-Diagnosis.pdf
└── MRI_MODELS/
    └── MRI_Classification_Models/
        ├── cnn_models/                  # Custom CNN architectures
        ├── pretrained_models/           # ResNet50, ResNet101, EfficientNet-B3
        ├── utils/                       # Data splitting, metrics, callbacks
        └── main.py                      # Central training/evaluation runner
```

---

## License

Provided as-is for research and educational purposes.
