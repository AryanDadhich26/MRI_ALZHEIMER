
MRI_ALZHEIMER
Deep learning based Alzheimer's Disease diagnosis using neuroimaging data. Final year mini project by Aryan Dadhich and Aviral Saini, LNMIIT, under guidance of Dr. Rahul Sharma.
Overview
This project builds a full pipeline to classify brain MRI scans into four stages of Alzheimer's progression, Cognitively Normal (CN), Early Mild Cognitive Impairment (EMCI), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD), using structural MRI data from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.
The pipeline covers two main stages, preprocessing raw DICOM scans into clean, standardized MRI slices, and training/evaluating several CNN and transfer learning architectures on those slices.
Preprocessing pipeline (preprocessing.py)
Raw DICOM scans go through the following steps before they're usable for training.
Conversion from DICOM series to SimpleITK image format, then to float32.
Denoising using SimpleITK's CurvatureFlow filter.
Skull stripping through ANTsPyNet's brain extraction model.
N4 bias field correction to fix scanner related intensity inconsistencies.
Spatial registration to the MNI-152 anatomical template using SyN registration through ANTsPy.
Reorientation to RAS coordinate space.
TorchIO based normalization, resampling to 1mm isotropic voxels, Z-normalization, crop or pad to 224x224x224, intensity rescaling, and flip augmentation.
Extraction of 15 centered slices each from the axial, coronal, and sagittal planes per scan.
This produced a curated dataset of 18,030 MRI slices, split evenly across the four diagnostic classes.
Model training (MRI_MODELS/MRI_Classification_Models)
The framework supports two families of models.
Custom CNNs, 5 layer, 10 layer, and 15 layer convolutional networks built from scratch.
Pretrained backbones, ResNet50, ResNet101, and EfficientNet-B3, all pretrained on ImageNet and fine-tuned on the MRI slices.
Training features include class weighting for imbalance, early stopping, learning rate scheduling, deterministic seeding for reproducibility, and automatic evaluation with confusion matrices, ROC curves, and classification reports.
Run training with a command like this.
python main.py --family pretrained --model resnet101 --epochs 200
Results
Custom CNNs (5, 10, 15 layer) achieved accuracies between 31% and 59%.
EfficientNet-B3 reached 52% accuracy with a validation AUC of 0.80.
ResNet50 reached 50% accuracy.
ResNet101 performed best, reaching 91% accuracy, a macro F1-score of 0.91, and per-class AUC values between 0.98 and 0.99.
The results show a clear gap between shallow custom CNNs and deeper transfer learning models, with ResNet101's residual connections and stable gradient flow giving it a strong edge for this kind of volumetric medical imaging task.
Full details, architecture diagrams, and per-model metrics are documented in the project report PDF included in the repository.
Tech stack
Python, TensorFlow, Keras, PyTorch, TorchIO, ANTsPy, ANTsPyNet, SimpleITK, NiBabel, Scikit-learn, Keras Tuner.
