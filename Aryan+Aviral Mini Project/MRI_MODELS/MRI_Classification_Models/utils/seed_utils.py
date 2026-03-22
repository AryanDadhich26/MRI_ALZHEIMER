"""
Random seed utilities for reproducibility.
"""

import os
import random
import numpy as np
import tensorflow as tf


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, and TensorFlow.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Ensure deterministic operations in TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Random seed set to {seed} for reproducibility")

