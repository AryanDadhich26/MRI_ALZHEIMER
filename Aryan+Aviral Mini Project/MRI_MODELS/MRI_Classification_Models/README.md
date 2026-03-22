# Alzheimer's Disease MRI Classification (Multi-Model Framework)
A comprehensive deep learning framework for classifying MRI brain images into four categories: **AD (Alzheimer's Disease)**, **MCI (Mild Cognitive Impairment)**, **CN (Cognitively Normal)**, and **EMCI (Early Mild Cognitive Impairment)**.

## 📁 Project Structure

```
MRI_Classification_Models/
├── cnn_models/
│   ├── cnn_5layer.py            # Custom 5-layer CNN
│   ├── cnn_10layer.py           # Deeper 10-layer CNN
│   ├── cnn_train_val_test.py   # Shared training pipeline
│   └── cnn_hyperparam_tuning.py # Hyperparameter tuning
│
├── pretrained_models/
│   ├── resnet50_imagenet.py     # ResNet50 pretrained on ImageNet
│   ├── resnet101_imagenet.py    # ResNet101 pretrained on ImageNet
│   ├── efficientnet_b3.py       # EfficientNetB3 pretrained backbone
│   ├── pretrained_train_val_test.py
│   └── pretrained_tuning.py
│
│  # Transformer models removed from this workspace
│  # If you need them, re-add a `transformer_models/` folder and update `main.py`
│
├── utils/
│   ├── data_split.py            # Dataset splitting (80-10-10)
│   ├── metrics_visualization.py  # Confusion matrix, ROC curves, etc.
│   ├── callbacks.py             # EarlyStopping, LR schedulers, etc.
│   ├── class_weights.py         # Class weight computation
│   └── seed_utils.py            # Random seed utilities
│
└── main.py                      # Central runner script
```

## 🚀 Quick Start

### 1. Dataset Preparation

Ensure your dataset follows this structure:
```
axial_pics/
├── AD/
├── MCI/
├── CN/
└── EMCI/
```

### 2. Split Dataset

```python
from utils.data_split import split_dataset_with_balance

train_counts, val_counts, test_counts = split_dataset_with_balance(
    original_data_dir="axial_pics",
    split_base_dir="axial_pics_splitt",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### Run split from terminal

You can run the dataset split directly from the command line (useful for automation):

```powershell
python -c "from utils.data_split import split_dataset_with_balance; split_dataset_with_balance(original_data_dir='axial_pics', split_base_dir='axial_pics_splitt', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)"
```

### 3. Train a Model

#### Using main.py (Recommended)

```bash
# Train 5-layer CNN
python main.py --family cnn --model cnn_5layer --epochs 200

# Train ResNet101
python main.py --family pretrained --model resnet101 --epochs 200

# Vision Transformer support removed (not included in this workspace)
```

#### Using Individual Scripts

```python
# Example: Train ResNet101
from pretrained_models.pretrained_train_val_test import run_full_pipeline
from pretrained_models.resnet101_imagenet import build_model, compile_model

def model_builder(num_classes=4, learning_rate=1e-5, optimizer='adam', **kwargs):
    model = build_model(num_classes=num_classes, **kwargs)
    return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)

model, history, metrics = run_full_pipeline(
    model_builder=model_builder,
    model_name="resnet101",
    epochs=200
)
```

### 4. Hyperparameter Tuning

```bash
# Tune CNN hyperparameters
python main.py --family cnn --model cnn_5layer --tune --trials 20

# Tune pretrained model hyperparameters
python main.py --family pretrained --model resnet50 --tune --trials 20
```

## 🧠 Available Models

### CNN Models
- **cnn_5layer**: Custom 5-layer CNN with Conv2D, BatchNorm, MaxPooling, Dropout
- **cnn_10layer**: Deeper 10-layer CNN architecture
- **cnn_15layer**: Deeper 15-conv-layer variant (stacked conv blocks)

### Pretrained Models
- **resnet50**: ResNet50 pretrained on ImageNet
- **resnet101**: ResNet101 pretrained on ImageNet
- **efficientnet_b3**: EfficientNetB3 pretrained backbone

### Transformer Models
Transformer models are not included in this workspace (removed). To re-add them,
place their implementations under a `transformer_models/` folder and update `main.py`.

## ⚙️ Features

### Training Pipeline
- ✅ Medically-safe data augmentation
- ✅ Deterministic training (reproducible results)
- ✅ Class weights for imbalanced datasets
- ✅ Early stopping and learning rate scheduling
- ✅ Model checkpointing
- ✅ GPU acceleration support

### Evaluation Metrics
- Accuracy (primary monitoring and checkpointing metric)
- Loss
- Precision, Recall, F1-score
- Confusion matrix (normalized by percentage)
- ROC curves and AUC (computed for analysis; training/checkpoints prioritize accuracy)
- Classification reports

### Hyperparameter Tuning
- Learning rate optimization
- Dropout rate tuning
- Filter size optimization (for CNNs)
- Layer freezing optimization (for pretrained models)
- Optimizer selection (Adam, RMSProp)
- Dense layer units tuning

## 📊 Output Files

After training, each model's outputs are stored in its own folder under `saved_models/{model_name}/`:
- `saved_models/{model_name}/best_{model_name}.h5` - Best model weights
- `saved_models/{model_name}/{model_name}_training_curves.png` - Training/validation curves
- `saved_models/{model_name}/confusionmatrix_{model_name}.png` - Confusion matrix visualization
- `saved_models/{model_name}/roc_curves_{model_name}.png` - ROC curves visualization
- `saved_models/{model_name}/classification_report_{model_name}.txt` - Text classification report
- `saved_models/{model_name}/metrics.csv` - Aggregated metrics CSV (one row per model_name, latest run replaces previous)
- `saved_models/{model_name}/logs/` - TensorBoard logs

## 🔧 Configuration

### Default Parameters
- **Image size**: 224x224
- **Batch size**: 32
- **Learning rate**: 
  - CNN: 1e-4
  - Pretrained: 1e-5
  - Transformer: 1e-4
- **Epochs**: 100
- **Class names**: ['AD', 'MCI', 'CN', 'EMCI']

### Customization

You can customize training parameters via command-line arguments or by modifying the scripts directly.

```bash
python main.py \
  --family pretrained \
  --model resnet101 \
  --split_dir "axial_pics_splitt" \
  --image_size 224 224 \
  --batch_size 32 \
    --epochs 200 \
  --learning_rate 1e-5 \
  --optimizer adam
```

## 📦 Dependencies

```bash
pip install tensorflow>=2.10.0
pip install keras-tuner
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install numpy
```

## 🎯 Usage Examples

### Example 1: Train and Evaluate ResNet101

```python
from pretrained_models.pretrained_train_val_test import run_full_pipeline
from pretrained_models.resnet101_imagenet import build_model, compile_model

def model_builder(num_classes=4, learning_rate=1e-5, optimizer='adam', **kwargs):
    model = build_model(num_classes=num_classes, **kwargs)
    return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)

model, history, metrics = run_full_pipeline(
    model_builder=model_builder,
    model_name="resnet101",
    epochs=100
)
```

### Example 2: Hyperparameter Tuning

```python
from cnn_models.cnn_hyperparam_tuning import tune_cnn_hyperparameters

best_hps, best_model = tune_cnn_hyperparameters(
    model_type='5layer',
    max_epochs=200,
    trials=20
)

print("Best hyperparameters:", best_hps.values)
```

### Example 3: Evaluate Saved Model

```python
from tensorflow.keras.models import load_model
from utils.metrics_visualization import evaluate_model
from pretrained_models.pretrained_train_val_test import get_data_generators

# Load model from its model-specific folder
model = load_model("saved_models/resnet101/best_resnet101.h5")

# Get test generator
_, _, test_gen = get_data_generators(split_base_dir='axial_pics_splitt')

# Evaluate (results will be saved into `saved_models/resnet101/`)
metrics = evaluate_model(
    model=model,
    generator=test_gen,
    class_names=['AD', 'MCI', 'CN', 'EMCI'],
    save_dir="saved_models/resnet101"
)
```

## 🔬 Model Architecture Details

### CNN Models
- Convolutional layers with BatchNormalization
- MaxPooling for downsampling
- Dropout for regularization
- L2 weight decay
- Dense classification head

### Pretrained Models
- ImageNet-pretrained backbones
- Custom classification head
- Fine-tuning of top layers
- GlobalAveragePooling2D
- Dropout and L2 regularization

### Transformer Models
- Patch-based or window-based attention
- Multi-head self-attention
- Positional embeddings
- MLP heads for classification
- Layer normalization

## 📝 Notes

- All models use medically-safe augmentation (conservative rotation, minimal shifts)
- Class weights are automatically computed to handle imbalanced datasets
- Training is deterministic (seed=42) for reproducibility
- GPU acceleration is automatically used if available
-- Models are saved with best validation accuracy

## One-line commands (quick)
Run these from the project root. Each line trains the model and saves outputs in `saved_models/{model_name}`.

- Train 5-layer CNN:
    ```powershell
    python main.py --family cnn --model cnn_5layer --epochs 100 --batch_size 32
    ```
- Train 10-layer CNN:
    ```powershell
    python main.py --family cnn --model cnn_10layer --epochs 100 --batch_size 32
    ```
- Train 15-layer CNN:
    ```powershell
    python main.py --family cnn --model cnn_15layer --epochs 100 --batch_size 32
    ```
- Train ResNet50 (pretrained):
    ```powershell
    python main.py --family pretrained --model resnet50 --epochs 100 --batch_size 16
    ```
- Train ResNet101 (pretrained):
    ```powershell
    python main.py --family pretrained --model resnet101 --epochs 100 --batch_size 16
    ```
- Train EfficientNetB3 (pretrained):
    ```powershell
    python main.py --family pretrained --model efficientnet_b3 --epochs 100 --batch_size 16
    ```

-- Hyperparameter tuning (example):
    ```powershell
    python main.py --family cnn --model cnn_5layer --tune --trials 20 --max_epochs 50
    ```

-- Evaluate a saved model (example):
    ```powershell
    python -c "from tensorflow.keras.models import load_model; from utils.metrics_visualization import evaluate_model; from pretrained_models.pretrained_train_val_test import get_data_generators; m=load_model('saved_models/resnet101/best_resnet101.h5'); _,_,tg=get_data_generators(); print(evaluate_model(m,tg,['AD','MCI','CN','EMCI'],'saved_models/resnet101'))"
    ```

## Recommendations to improve accuracy (next steps)
These are practical experiments you can try (order by expected impact):

- Increase effective dataset size / quality: collect more labeled MRIs, or use stronger data augmentation (elastic deformations, small rotations up to 10-15°, brightness/contrast, random crops). More data helps most.
- Fine-tune more of the pretrained backbone: reduce `freeze_layers` to unfreeze more layers (careful with small learning rate, e.g., 1e-5).
- Use a learning rate schedule: cosine decay, ReduceLROnPlateau (already present), or cyclical learning rates. Try a short warmup then lower LR.
- Run longer training with early stopping off for more epochs (monitor validation accuracy). Increase epochs to 200-300 if compute allows.
- Use stratified sampling and stronger class balancing (oversample minority classes, or use focal loss if class imbalance persists).
- Try regularization tweaks: reduce label_smoothing (or remove), reduce L2 weight decay if underfitting, or increase dropout if overfitting.
- Use advanced augmentation / Mixup / CutMix to improve generalization.
- Try image preprocessing specific to MRI (intensity normalization, histogram equalization, CLAHE) before feeding images.
- Use larger input size (e.g., 256x256) if memory allows — pretrained models often benefit.
- Experiment with optimizer settings: AdamW, lookahead, or SGD with momentum for fine-tuning.
- Use ensemble of top models (e.g., average predictions from ResNet50 + EfficientNet + best CNN) for final accuracy boost.
- Use a learning rate finder (e.g., fastai's LR finder) to choose a better initial LR.

If you want, I can: add a small `scripts/` CLI wrapper for training/evaluation, implement Mixup/CutMix augmentation, or set up a hyperparameter sweep config (e.g., using KerasTuner or Optuna) focused on learning rate, weight decay, and dense units.

## 🤝 Contributing

Feel free to extend this framework with additional models or features!

## 📄 License

This project is provided as-is for research and educational purposes.

