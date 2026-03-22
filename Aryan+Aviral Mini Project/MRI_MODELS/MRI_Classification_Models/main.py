"""
Main runner script for MRI Classification Models.
Allows selection of model family (CNN / Pretrained / Transformer) and runs training/validation/testing.
"""

import os
import sys
import argparse
import tensorflow as tf

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.seed_utils import set_random_seed


def run_cnn_model(model_name, **kwargs):
    """Run CNN model training."""
    from cnn_models.cnn_train_val_test import run_full_pipeline
    
    if model_name == 'cnn_5layer':
        from cnn_models.cnn_5layer import build_model, compile_model
        
        def model_builder(num_classes=4, learning_rate=1e-4, optimizer='adam', **model_kwargs):
            model = build_model(num_classes=num_classes, **model_kwargs)
            return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)
        
        return run_full_pipeline(
            model_builder=model_builder,
            model_name="cnn_5layer",
            **kwargs
        )
    
    elif model_name == 'cnn_10layer':
        from cnn_models.cnn_10layer import build_model, compile_model
        
        def model_builder(num_classes=4, learning_rate=1e-4, optimizer='adam', **model_kwargs):
            model = build_model(num_classes=num_classes, **model_kwargs)
            return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)
        
        return run_full_pipeline(
            model_builder=model_builder,
            model_name="cnn_10layer",
            **kwargs
        )
    elif model_name == 'cnn_15layer' or model_name == 'cnn_15':
        from cnn_models.cnn_15layer import build_model, compile_model

        def model_builder(num_classes=4, learning_rate=1e-4, optimizer='adam', **model_kwargs):
            model = build_model(num_classes=num_classes, **model_kwargs)
            return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)

        return run_full_pipeline(
            model_builder=model_builder,
            model_name="cnn_15layer",
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown CNN model: {model_name}")


def run_pretrained_model(model_name, **kwargs):
    """Run pretrained model training."""
    from pretrained_models.pretrained_train_val_test import run_full_pipeline
    
    if model_name == 'resnet50':
        from pretrained_models.resnet50_imagenet import build_model, compile_model
        
        def model_builder(num_classes=4, learning_rate=1e-5, optimizer='adam', **model_kwargs):
            model = build_model(num_classes=num_classes, **model_kwargs)
            return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)
        
        return run_full_pipeline(
            model_builder=model_builder,
            model_name="resnet50",
            **kwargs
        )
    
    elif model_name == 'resnet101':
        from pretrained_models.resnet101_imagenet import build_model, compile_model
        
        def model_builder(num_classes=4, learning_rate=1e-5, optimizer='adam', **model_kwargs):
            model = build_model(num_classes=num_classes, **model_kwargs)
            return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)
        
        return run_full_pipeline(
            model_builder=model_builder,
            model_name="resnet101",
            **kwargs
        )
    
    elif model_name == 'efficientnet_b3':
        from pretrained_models.efficientnet_b3 import build_model, compile_model
        
        def model_builder(num_classes=4, learning_rate=1e-5, optimizer='adam', **model_kwargs):
            model = build_model(num_classes=num_classes, **model_kwargs)
            return compile_model(model, learning_rate=learning_rate, optimizer=optimizer)
        
        return run_full_pipeline(
            model_builder=model_builder,
            model_name="efficientnet_b3",
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown pretrained model: {model_name}")


## Transformer models support removed



def run_hyperparameter_tuning(model_family, model_name, **kwargs):
    """Run hyperparameter tuning for a model."""
    if model_family == 'cnn':
        from cnn_models.cnn_hyperparam_tuning import tune_cnn_hyperparameters
        return tune_cnn_hyperparameters(model_type=model_name, **kwargs)
    
    elif model_family == 'pretrained':
        from pretrained_models.pretrained_tuning import tune_pretrained_hyperparameters
        return tune_pretrained_hyperparameters(model_type=model_name, **kwargs)
    
    # transformer family removed
    
    else:
        raise ValueError(f"Unknown model family: {model_family}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MRI Classification Models - Main Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 5-layer CNN
  python main.py --family cnn --model cnn_5layer --epochs 100
  
  # Train ResNet101
  python main.py --family pretrained --model resnet101 --epochs 100
  
    
  # Hyperparameter tuning
  python main.py --family cnn --model cnn_5layer --tune --trials 20
        """
    )
    
    parser.add_argument(
        '--family',
        type=str,
        required=True,
        choices=['cnn', 'pretrained'],
        help='Model family: cnn or pretrained'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., cnn_5layer, resnet50, vit)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run hyperparameter tuning instead of training'
    )
    
    parser.add_argument(
        '--split_dir',
        type=str,
        default='axial_pics_splitt',
        help='Base directory for split dataset (default: axial_pics_splitt)'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Image size as height width (default: 224 224)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs (default: 100)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (default: model-specific)'
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'rmsprop'],
        help='Optimizer (default: adam)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help='Number of trials for hyperparameter tuning (default: 20)'
    )
    
    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        default=['AD', 'MCI', 'CN', 'EMCI'],
        help='Class names (default: AD MCI CN EMCI)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(42)
    
    # Prepare common arguments
    common_kwargs = {
        'split_base_dir': args.split_dir,
        'image_size': tuple(args.image_size),
        'batch_size': args.batch_size,
        'class_names': args.class_names,
    }
    
    if args.tune:
        # Hyperparameter tuning
        common_kwargs.update({
            'max_epochs': args.epochs,
            'trials': args.trials,
        })
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING: {args.family.upper()} - {args.model}")
        print(f"{'='*60}\n")
        
        best_hps, best_model = run_hyperparameter_tuning(
            model_family=args.family,
            model_name=args.model,
            **common_kwargs
        )
        
        print("\n✅ Hyperparameter tuning completed!")
        print(f"Best model saved. Use the best hyperparameters to retrain.")
    
    else:
        # Training
        if args.learning_rate:
            common_kwargs['learning_rate'] = args.learning_rate
        common_kwargs['optimizer'] = args.optimizer
        common_kwargs['epochs'] = args.epochs
        
        print(f"\n{'='*60}")
        print(f"TRAINING: {args.family.upper()} - {args.model}")
        print(f"{'='*60}\n")
        
        if args.family == 'cnn':
            model, history, metrics = run_cnn_model(
                model_name=args.model,
                **common_kwargs
            )
        elif args.family == 'pretrained':
            model, history, metrics = run_pretrained_model(
                model_name=args.model,
                **common_kwargs
            )
        # transformer family removed
        
        print("\n✅ Training completed!")
        print(f"Model saved to: saved_models/{args.model}/best_{args.model}.h5")
        print(f"Metrics and plots saved to: saved_models/{args.model}/")


if __name__ == "__main__":
    main()

