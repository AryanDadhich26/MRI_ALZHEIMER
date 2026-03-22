"""
Custom 10-layer CNN for MRI brain image classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D,
    Dropout, Flatten, Dense
)
from tensorflow.keras import regularizers


def build_model(
    input_shape=(224, 224, 3),
    num_classes=4,
    filters=[48, 96, 96, 160, 160, 256, 256, 384, 384, 512, 656],
    # very small dropout decrease
    dropout_rate=0.28,
    l2_reg=1e-5,
    dense_units=768
):
    """
    Build a 10-layer CNN model.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        filters: List of filter sizes for each conv layer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization coefficient
        dense_units: Number of units in dense layer
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Layer 1
        Conv2D(
            filters[0], (3, 3), activation='relu',
            input_shape=input_shape,
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        
        # Layer 2
        Conv2D(
            filters[1], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Layer 3
        Conv2D(
            filters[2], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        
        # Layer 4
        Conv2D(
            filters[3], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Layer 5
        Conv2D(
            filters[4], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        
        # Layer 6
        Conv2D(
            filters[5], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Layer 7
        Conv2D(
            filters[6], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        
        # Layer 8
        Conv2D(
            filters[7], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Layer 9
        Conv2D(
            filters[8], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        
        # Layer 10
        Conv2D(
            filters[9], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Layer 11 (small extra conv block)
        Conv2D(
            filters[10], (3, 3), activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),

        # Classification head
        Flatten(),
        Dense(
            dense_units, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(
    model,
    learning_rate=1e-4,
    optimizer='adam',
    label_smoothing=0.1
):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ('adam' or 'rmsprop')
        label_smoothing: Label smoothing factor
        
    Returns:
        Compiled model
    """
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    model = build_model()
    model = compile_model(model)
    model.summary()

