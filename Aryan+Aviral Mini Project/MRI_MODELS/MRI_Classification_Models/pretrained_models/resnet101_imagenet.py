"""
ResNet101 pretrained on ImageNet for MRI brain image classification.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def build_model(
    input_shape=(224, 224, 3),
    num_classes=4,
    # tiny dropout decrease and slightly larger head
    dropout_rate=0.39,
    dense_units=704,
    l2_reg=1e-4,
    # unfreeze 3 more layers for light fine-tuning
    freeze_layers=110
):
    """
    Build ResNet101 model with custom classification head.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        dense_units: Number of units in dense layer
        l2_reg: L2 regularization coefficient
        freeze_layers: Number of layers to freeze (default: 160)
        
    Returns:
        Keras model
    """
    # Load pretrained ResNet101
    base_model = ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze layers
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    
    # Unfreeze remaining layers for fine-tuning
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
    
    return model


def compile_model(
    model,
    learning_rate=1e-5,
    optimizer='adam',
    label_smoothing=0.2
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

