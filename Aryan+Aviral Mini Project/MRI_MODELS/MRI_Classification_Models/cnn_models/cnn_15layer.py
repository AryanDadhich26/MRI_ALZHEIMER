"""
Simple 15-layer CNN for MRI classification (basic deeper variant).
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
    base_filters=48,
    dropout_rate=0.3,
    l2_reg=1e-5,
    dense_units=768
):
    """
    Build a simple 15-layer CNN (stacked conv blocks).
    This is intentionally straightforward: repeated conv - bn - pool - dropout blocks.
    """
    filters = [base_filters * (2 ** i) for i in range(5)]  # 48, 96, 192, 384, 768

    model = Sequential()

    # 3 conv blocks per filter size (5*3 = 15 conv layers)
    for f in filters:
        for _ in range(3):
            model.add(Conv2D(
                f, (3, 3), activation='relu', padding='same',
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

    # Removed extra small conv block (this made it 16 layers)
    # Everything else remains EXACTLY the same

    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, learning_rate=1e-4, optimizer='adam', label_smoothing=0.1):
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


if __name__ == '__main__':
    m = build_model()
    m = compile_model(m)
    m.summary()
