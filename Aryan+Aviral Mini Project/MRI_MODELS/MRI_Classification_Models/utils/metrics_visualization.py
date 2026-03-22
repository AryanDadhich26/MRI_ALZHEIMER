"""
Metrics and visualization utilities for model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from itertools import cycle


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # AUC plot (if available) or Loss plot otherwise
    plt.subplot(1, 2, 2)
    if 'auc' in history.history or 'val_auc' in history.history:
        if 'auc' in history.history:
            plt.plot(history.history['auc'], label='Train AUC', linewidth=2)
        if 'val_auc' in history.history:
            plt.plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
        plt.title('Model AUC', fontsize=14, fontweight='bold')
        plt.ylabel('AUC', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        if 'loss' in history.history:
            plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, save_path=None):
    """
    Plot confusion matrix with percentage normalization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize by row (percentage)
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        title = 'Confusion Matrix (% of Actual Class)'
        annot = cm_percent
    else:
        cm_percent = cm
        fmt = 'd'
        title = 'Confusion Matrix (Count)'
        annot = cm
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        # Ensure parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true.ravel(), y_pred_proba.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    plt.plot(
        fpr["micro"], tpr["micro"],
        color='deeppink', linestyle='--', lw=2,
        label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})'
    )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print(report_str)
    return report_str
    print("="*60)


def evaluate_model(model, generator, class_names, save_dir=None):
    """
    Comprehensive model evaluation with all metrics and visualizations.
    
    Args:
        model: Trained Keras model
        generator: Data generator for evaluation
        class_names: List of class names
        save_dir: Optional directory to save plots
        
    Returns:
        Dictionary with evaluation metrics
    """
    generator.reset()
    
    # Evaluate
    results = model.evaluate(generator, verbose=1)
    metric_names = model.metrics_names

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    # Print each metric but highlight accuracy + auc together when present
    metrics_map = dict(zip(metric_names, results))
    for name, value in metrics_map.items():
        print(f"{name}: {value:.4f}")
    # Explicit short summary
    acc = metrics_map.get('accuracy') or metrics_map.get('acc')
    auc_val = metrics_map.get('auc')
    summary_parts = []
    if acc is not None:
        summary_parts.append(f"Accuracy={acc:.4f}")
    if auc_val is not None:
        summary_parts.append(f"AUC={auc_val:.4f}")
    if summary_parts:
        print(" | ".join(summary_parts))
    
    # Predictions
    generator.reset()
    y_pred_proba = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = generator.classes
    
    # Convert to one-hot for ROC
    y_true_onehot = np.zeros((len(y_true), len(class_names)))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    # Classification report (print and capture). Also obtain dict for CSV.
    try:
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        # Also keep the printable string
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        print(report_str)
    except Exception:
        report_dict = None
        report_str = print_classification_report(y_true, y_pred, class_names)
    
    # Prepare save directory (default to ./results)
    base_save_dir = save_dir if save_dir else 'results'
    Path(base_save_dir).mkdir(parents=True, exist_ok=True)
    model_name = getattr(model, 'name', 'model')
    cm_path = os.path.join(base_save_dir, f"confusionmatrix_{model_name}.png")
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=cm_path
    )

    # Save classification report to file next to confusion matrix
    cr_path = os.path.join(base_save_dir, f"classification_report_{model_name}.txt")
    try:
        with open(cr_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"✅ Classification report saved to {cr_path}")
    except Exception as e:
        print(f"⚠️ Failed to save classification report: {e}")
    
    # ROC curves
    plot_roc_curves(
        y_true_onehot, y_pred_proba, class_names,
        save_path=os.path.join(base_save_dir, f"roc_curves_{model_name}.png")
    )
    
    # Calculate additional metrics
    metrics = {
        'loss': results[0],
        'accuracy': results[1] if len(results) > 1 else None,
        'auc': results[2] if len(results) > 2 else None,
    }
    
    # Per-class AUC
    try:
        per_class_auc = roc_auc_score(
            y_true_onehot, y_pred_proba, average=None
        )
        metrics['per_class_auc'] = dict(zip(class_names, per_class_auc))
        print("\nPer-class AUC:")
        for class_name, auc_score in metrics['per_class_auc'].items():
            print(f"  {class_name}: {auc_score:.4f}")
    except:
        pass
    
    # Save metrics to CSV (replace existing entry for the same model_name)
    try:
        save_metrics_to_csv(
            model_name=model_name,
            metrics=metrics,
            classification_report_dict=report_dict,
            per_class_auc=metrics.get('per_class_auc', {}),
            class_names=class_names,
            csv_path=os.path.join(base_save_dir, 'metrics.csv')
        )
    except Exception as e:
        print(f"⚠️ Failed to save metrics to CSV: {e}")
    
    return metrics


def save_metrics_to_csv(
    model_name,
    metrics,
    classification_report_dict,
    per_class_auc,
    class_names,
    csv_path="saved_models/metrics.csv"
):
    """
    Save or append model evaluation metrics to a CSV. If an entry for the same
    `model_name` exists, it will be replaced with the new one.

    Columns include basic metrics (loss, accuracy, auc), per-class precision/recall/f1/support,
    macro/weighted averages, and per-class AUCs.
    """
    # Build the row dictionary
    row = {
        'model_name': model_name,
        'timestamp': datetime.utcnow().isoformat(),
        'loss': metrics.get('loss'),
        'accuracy': metrics.get('accuracy'),
        'auc': metrics.get('auc')
    }

    # Add per-class precision/recall/f1/support if available
    if classification_report_dict:
        for cls in class_names:
            cls_metrics = classification_report_dict.get(cls, {})
            row[f'{cls}_precision'] = cls_metrics.get('precision')
            row[f'{cls}_recall'] = cls_metrics.get('recall')
            row[f'{cls}_f1'] = cls_metrics.get('f1-score')
            row[f'{cls}_support'] = cls_metrics.get('support')

        # Add macro and weighted averages
        for avg in ['macro avg', 'weighted avg']:
            avg_key = avg.replace(' ', '_')
            avg_metrics = classification_report_dict.get(avg, {})
            row[f'{avg_key}_precision'] = avg_metrics.get('precision')
            row[f'{avg_key}_recall'] = avg_metrics.get('recall')
            row[f'{avg_key}_f1'] = avg_metrics.get('f1-score')
            row[f'{avg_key}_support'] = avg_metrics.get('support')

    # Add per-class AUCs
    if per_class_auc:
        for cls in class_names:
            row[f'auc_{cls}'] = per_class_auc.get(cls)

    # Ensure directory exists
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # Read existing records (if any)
    existing_rows = []
    existing_fieldnames = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                existing_fieldnames = reader.fieldnames or []
                for r in reader:
                    existing_rows.append(r)
        except Exception:
            existing_rows = []

    # Remove any existing rows for this model_name
    existing_rows = [r for r in existing_rows if r.get('model_name') != model_name]

    # Combine fieldnames
    new_fieldnames = list(dict.fromkeys(
        (existing_fieldnames or []) + list(row.keys())
    ))

    # Normalize existing rows to new fieldnames
    for r in existing_rows:
        for fn in new_fieldnames:
            if fn not in r:
                r[fn] = ''

    # Add the new row (with string conversions)
    str_row = {k: ('' if v is None else v) for k, v in row.items()}
    existing_rows.append(str_row)

    # Write back to CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
        print(f"✅ Metrics saved to CSV: {csv_path}")
    except Exception as e:
        print(f"⚠️ Failed to write metrics CSV: {e}")

