"""
Class weight computation utility for handling imbalanced datasets.
"""

import numpy as np
from collections import Counter


def calculate_class_weights(generator, max_weight=3.0):
    """
    Calculate class weights for imbalanced dataset.
    Uses soft-capped weights (max multiplier) to prevent extreme weights.
    
    Args:
        generator: Keras ImageDataGenerator with classes attribute
        max_weight: Maximum weight multiplier (default: 3.0)
        
    Returns:
        Dictionary mapping class index to weight
    """
    counts = Counter(generator.classes)
    median = np.median(list(counts.values()))
    
    # Soft capped weights (max multiplier)
    class_weights = {
        cls: min(np.sqrt(median / count), max_weight)
        for cls, count in counts.items()
    }
    
    print("\nClass weights:")
    for cls, weight in sorted(class_weights.items()):
        print(f"  Class {cls}: weight = {weight:.2f}")
    
    return class_weights


def get_class_names_from_generator(generator):
    """Extract class names from generator."""
    return list(generator.class_indices.keys())

