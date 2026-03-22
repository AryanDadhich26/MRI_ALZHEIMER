"""
Shared training, validation, and testing pipeline for pretrained models.
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed_utils import set_random_seed
from utils.class_weights import calculate_class_weights
from utils.callbacks import get_default_callbacks
from utils.metrics_visualization import (
    plot_training_curves,
    evaluate_model
)


def get_data_generators(
    split_base_dir="axial_pics_splitt",
    image_size=(224, 224),
    batch_size=32,
    class_names=['AD', 'MCI', 'CN', 'EMCI']
):
    """
    Create data generators for training, validation, and testing.
    Uses medically-safe augmentation for training.
    
    Args:
        split_base_dir: Base directory containing train/val/test folders
        image_size: Target image size (height, width)
        batch_size: Batch size for training
        class_names: List of class names
        
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    train_dir = os.path.join(split_base_dir, "train")
    val_dir = os.path.join(split_base_dir, "val")
    test_dir = os.path.join(split_base_dir, "test")
    
    # Medically-constrained augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=8,  # Conservative rotation
        width_shift_range=0.04,  # Minimal shifts
        height_shift_range=0.04,
        zoom_range=0.08,  # Small zoom range
        brightness_range=[0.9, 1.1],  # Mild brightness variation
        horizontal_flip=True,  # Only horizontal flips
        fill_mode='constant',  # Black background fill
        cval=0
    )
    
    # No augmentation for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        classes=class_names
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=class_names
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=class_names
    )
    
    return train_gen, val_gen, test_gen


def train_model(
    model,
    train_gen,
    val_gen,
    epochs=200,
    class_weights=None,
    model_name="pretrained_model",
    save_dir="saved_models",
    callbacks=None
):
    """
    Train a pretrained model.
    
    Args:
        model: Compiled Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        epochs: Number of training epochs
        class_weights: Dictionary of class weights
        model_name: Name for saving model
        save_dir: Directory to save model
        callbacks: List of callbacks (if None, uses default)
        
    Returns:
        Training history
    """
    if callbacks is None:
        callbacks = get_default_callbacks(
            model_name=model_name,
            save_dir=save_dir
        )
    
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {train_gen.batch_size}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Count trainable vs frozen layers
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    total_count = len(model.layers)
    print(f"Trainable layers: {trainable_count}/{total_count}")
    print(f"{'='*60}\n")
    
    # Use GPU if available
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    
    return history


def run_full_pipeline(
    model_builder,
    model_name="pretrained_model",
    split_base_dir="axial_pics_splitt",
    image_size=(224, 224),
    batch_size=32,
    epochs=100,
    class_names=['AD', 'MCI', 'CN', 'EMCI'],
    learning_rate=1e-5,
    optimizer='adam',
    **model_kwargs
):
    """
    Run complete training, validation, and testing pipeline.
    
    Args:
        model_builder: Function that builds and compiles the model
        model_name: Name for the model
        split_base_dir: Base directory for split dataset
        image_size: Target image size
        batch_size: Batch size
        epochs: Number of epochs
        class_names: List of class names
        learning_rate: Learning rate
        optimizer: Optimizer type
        **model_kwargs: Additional arguments for model builder
        
    Returns:
        Tuple of (model, history, test_metrics)
    """
    # Set random seed
    set_random_seed(42)
    
    # Get data generators
    train_gen, val_gen, test_gen = get_data_generators(
        split_base_dir=split_base_dir,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)
    
    # Build and compile model
    model = model_builder(
        num_classes=len(class_names),
        learning_rate=learning_rate,
        optimizer=optimizer,
        **model_kwargs
    )
    
    # Train model
    history = train_model(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=epochs,
        class_weights=class_weights,
        model_name=model_name,
        save_dir="saved_models"
    )
    
    # Plot training curves and save into model-specific folder
    model_save_dir = os.path.join("saved_models", model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    plot_training_curves(
        history,
        save_path=os.path.join(model_save_dir, f"{model_name}_training_curves.png")
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    test_metrics = evaluate_model(
        model=model,
        generator=test_gen,
        class_names=class_names,
        save_dir=model_save_dir
    )
    
    return model, history, test_metrics


if __name__ == "__main__":
    # Example usage with ResNet101
    from resnet101_imagenet import build_model, compile_model
    
    def model_builder(num_classes=4, learning_rate=1e-5, optimizer='adam', **kwargs):
        model = build_model(num_classes=num_classes, **kwargs)
        return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)
    
    model, history, metrics = run_full_pipeline(
        model_builder=model_builder,
        model_name="resnet101",
        epochs=200
    )

