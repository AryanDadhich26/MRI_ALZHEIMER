"""
Data splitting utility for MRI classification dataset.
Handles train/val/test split with balanced class distribution.
"""

import os
import shutil
import random


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)


def check_class_balance(data_dir, class_names):
    """
    Check and display class distribution in a directory.
    
    Args:
        data_dir: Path to directory containing class subdirectories
        class_names: List of class names
        
    Returns:
        Dictionary with class counts
    """
    counts = {}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            counts[class_name] = len([
                f for f in os.listdir(class_dir) 
                if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ])
        else:
            counts[class_name] = 0
    
    print(f"\nClass distribution in {os.path.basename(data_dir)}:")
    total = sum(counts.values())
    for k, v in counts.items():
        percentage = (v/total*100) if total else 0
        print(f"  {k}: {v} samples ({percentage:.1f}%)")
    
    return counts


def split_dataset_with_balance(
    original_data_dir="axial_pics",
    split_base_dir="axial_pics_splitt",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    class_names=['AD', 'MCI', 'CN', 'EMCI'],
    seed=42
):
    """
    Split dataset into train/val/test with balanced class distribution.
    
    Args:
        original_data_dir: Path to original dataset directory
        split_base_dir: Base directory for split dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        class_names: List of class names
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_counts, val_counts, test_counts) dictionaries
    """
    set_random_seed(seed)
    
    train_dir = os.path.join(split_base_dir, "train")
    val_dir = os.path.join(split_base_dir, "val")
    test_dir = os.path.join(split_base_dir, "test")
    
    # Check if already split
    if all(os.path.exists(os.path.join(d, c)) 
           for d in [train_dir, val_dir, test_dir] 
           for c in class_names):
        print("✅ Dataset already split. Skipping...")
        return (
            check_class_balance(train_dir, class_names),
            check_class_balance(val_dir, class_names),
            check_class_balance(test_dir, class_names)
        )
    
    print("📂 Splitting dataset with balanced classes...")
    
    # Create directories
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
        for c in class_names:
            os.makedirs(os.path.join(d, c), exist_ok=True)
    
    # Split each class
    for class_name in class_names:
        src_dir = os.path.join(original_data_dir, class_name)
        if not os.path.exists(src_dir):
            print(f"⚠ Skipping {class_name}, source folder not found: {src_dir}")
            continue
        
        images = [
            f for f in os.listdir(src_dir) 
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]
        random.shuffle(images)
        
        total = len(images)
        if total == 0:
            print(f"⚠ No images found for class {class_name}")
            continue
        
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Copy to train
        for img in images[:train_end]:
            shutil.copy2(
                os.path.join(src_dir, img),
                os.path.join(train_dir, class_name, img)
            )
        
        # Copy to validation
        for img in images[train_end:val_end]:
            shutil.copy2(
                os.path.join(src_dir, img),
                os.path.join(val_dir, class_name, img)
            )
        
        # Copy to test
        for img in images[val_end:]:
            shutil.copy2(
                os.path.join(src_dir, img),
                os.path.join(test_dir, class_name, img)
            )
    
    print("✅ Split complete.")
    
    return (
        check_class_balance(train_dir, class_names),
        check_class_balance(val_dir, class_names),
        check_class_balance(test_dir, class_names)
    )


if __name__ == "__main__":
    # Example usage
    train_counts, val_counts, test_counts = split_dataset_with_balance()

