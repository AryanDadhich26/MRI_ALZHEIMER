"""
Hyperparameter tuning for pretrained models using KerasTuner.
"""

import os
import sys
import tensorflow as tf
import kerastuner as kt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed_utils import set_random_seed
from utils.class_weights import calculate_class_weights
from pretrained_train_val_test import get_data_generators


def build_tunable_resnet50(hp, input_shape=(224, 224, 3), num_classes=4):
    """
    Build tunable ResNet50 model.
    
    Args:
        hp: KerasTuner HyperParameters object
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    
    # Hyperparameters to tune
    lr = hp.Float('learning_rate', min_value=1e-6, max_value=1e-4, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.4, max_value=0.8, step=0.1)
    l2_reg = hp.Choice('l2_reg', values=[1e-5, 1e-4, 1e-3])
    dense_units = hp.Int('dense_units', min_value=256, max_value=1024, step=256)
    freeze_layers = hp.Int('freeze_layers', min_value=100, max_value=170, step=10)
    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    
    # Load pretrained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze layers
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    
    if dense_units > 0:
        x = Dense(
            dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


class PretrainedHyperModel(kt.HyperModel):
    """HyperModel wrapper for pretrained model tuning."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4, model_type='resnet50'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
    
    def build(self, hp):
        if self.model_type == 'resnet50':
            return build_tunable_resnet50(hp, self.input_shape, self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def tune_pretrained_hyperparameters(
    model_type='resnet50',
    split_base_dir="axial_pics_splitt",
    image_size=(224, 224),
    batch_size=32,
    class_names=['AD', 'MCI', 'CN', 'EMCI'],
    max_epochs=200,
    trials=20,
    project_name="pretrained_tuning"
):
    """
    Tune hyperparameters for pretrained models.
    
    Args:
        model_type: Type of pretrained model ('resnet50', 'resnet101', 'efficientnet_b3')
        split_base_dir: Base directory for split dataset
        image_size: Target image size
        batch_size: Batch size
        class_names: List of class names
        max_epochs: Maximum epochs per trial
        trials: Number of trials
        project_name: Name for tuning project
        
    Returns:
        Best hyperparameters and best model
    """
    set_random_seed(42)
    
    # Get data generators
    train_gen, val_gen, _ = get_data_generators(
        split_base_dir=split_base_dir,
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_gen)
    
    # Create hypermodel
    hypermodel = PretrainedHyperModel(
        input_shape=(*image_size, 3),
        num_classes=len(class_names),
        model_type=model_type
    )
    
    # Create tuner
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=trials,
        directory="tuning_results",
        project_name=project_name
    )
    
    # Search for best hyperparameters
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Trials: {trials}")
    print(f"Max epochs per trial: {max_epochs}")
    print(f"{'='*60}\n")
    
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        tuner.search(
            train_gen,
            epochs=max_epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            verbose=1
        )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n" + "="*60)
    print("BEST HYPERPARAMETERS")
    print("="*60)
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    print("="*60)
    
    # Build best model
    best_model = tuner.hypermodel.build(best_hps)
    
    return best_hps, best_model


if __name__ == "__main__":
    # Example usage
    best_hps, best_model = tune_pretrained_hyperparameters(
        model_type='resnet50',
        max_epochs=200,
        trials=10
    )
    
    print("\nBest model summary:")
    best_model.summary()

