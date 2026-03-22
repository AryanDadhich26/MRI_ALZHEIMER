"""
Custom callbacks and callback utilities for model training.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)


def get_default_callbacks(
    model_name="model",
    monitor="val_accuracy",
    save_dir="saved_models",
    early_stopping_patience=15,
    lr_patience=5,
    min_lr=1e-7,
    restore_best_weights=True
):
    """
    Create default callbacks for training.
    
    Args:
        model_name: Name for saved model file
        monitor: Metric to monitor (default: 'val_accuracy')
        save_dir: Directory to save model checkpoints
        early_stopping_patience: Patience for early stopping
        lr_patience: Patience for learning rate reduction
        min_lr: Minimum learning rate
        restore_best_weights: Whether to restore best weights on early stop
        
    Returns:
        List of callbacks
    """
    # Create per-model directory so each model has its own artifacts folder
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            mode='max' if 'auc' in monitor or 'accuracy' in monitor else 'min',
            restore_best_weights=restore_best_weights,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.2,
            patience=lr_patience,
            min_lr=min_lr,
            mode='max' if 'auc' in monitor or 'accuracy' in monitor else 'min',
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_save_dir, f"best_{model_name}.h5"),
            monitor=monitor,
            save_best_only=True,
            mode='max' if 'auc' in monitor or 'accuracy' in monitor else 'min',
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(model_save_dir, "logs", model_name),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks


def get_custom_callbacks(
    callbacks_list=None,
    model_name="model",
    save_dir="saved_models"
):
    """
    Create custom callbacks from a list.
    
    Args:
        callbacks_list: List of callback configurations
        model_name: Name for saved model file
        save_dir: Directory to save model checkpoints
        
    Returns:
        List of callbacks
    """
    if callbacks_list is None:
        return get_default_callbacks(model_name=model_name, save_dir=save_dir)
    
    callbacks = []
    os.makedirs(save_dir, exist_ok=True)
    
    for cb_config in callbacks_list:
        cb_type = cb_config.get('type', '')
        
        if cb_type == 'early_stopping':
            callbacks.append(EarlyStopping(**cb_config.get('params', {})))
        elif cb_type == 'reduce_lr':
            callbacks.append(ReduceLROnPlateau(**cb_config.get('params', {})))
        elif cb_type == 'checkpoint':
            params = cb_config.get('params', {})
            if 'filepath' not in params:
                params['filepath'] = os.path.join(
                    save_dir, f"best_{model_name}.h5"
                )
            callbacks.append(ModelCheckpoint(**params))
        elif cb_type == 'tensorboard':
            params = cb_config.get('params', {})
            if 'log_dir' not in params:
                params['log_dir'] = os.path.join(
                    save_dir, "logs", model_name
                )
            callbacks.append(TensorBoard(**params))
    
    return callbacks

