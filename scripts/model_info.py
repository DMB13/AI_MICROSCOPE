"""Model information and utility script for AI_MICROSCOPE.

Provides detailed information about the trained model, class mappings,
and utilities for testing and debugging.

Run from project root:
    python scripts/model_info.py
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def show_model_info():
    """Display detailed model information."""
    print("\n" + "=" * 70)
    print("MODEL INFORMATION".center(70))
    print("=" * 70)
    
    from inference.inference import load_model
    from model.model_config import (
        MODEL_PATH, MODEL_INPUT_SIZE, find_model_file, load_class_indices
    )
    
    # File information
    try:
        model_file = find_model_file()
        size_mb = model_file.stat().st_size / 1024 / 1024
        print(f"\nFile Information:")
        print(f"  Path: {model_file}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Format: Keras (.keras)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Model architecture
    try:
        model = load_model()
        print(f"\nModel Architecture:")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Number of layers: {len(model.layers)}")
        print(f"  Total parameters: {model.count_params():,}")
        
        # Count trainable vs non-trainable
        trainable_params = sum(1 for layer in model.layers if layer.trainable)
        print(f"  Trainable layers: {trainable_params}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Class information
    try:
        classes = load_class_indices()
        print(f"\nClass Information:")
        print(f"  Total classes: {len(classes)}")
        print(f"\n  Class Mapping:")
        
        # Show primary classes
        primary = {i: classes[i] for i in range(min(5, len(classes)))}
        for idx, name in primary.items():
            marker = " ← Primary" if idx < 3 else ""
            print(f"    {idx}: {name}{marker}")
        
        if len(classes) > 5:
            print(f"    ... ({len(classes) - 5} more classes)")
            
    except Exception as e:
        print(f"Error loading class indices: {e}")
        return
    
    # Preprocessing information
    print(f"\nPreprocessing Configuration:")
    print(f"  Input size: {MODEL_INPUT_SIZE[0]} × {MODEL_INPUT_SIZE[1]}")
    print(f"  Channels: RGB (3)")
    print(f"  Normalization: [0, 255] → [0, 1]")
    print(f"  Interpolation: Bilinear")


def show_layer_summary():
    """Show detailed layer summary."""
    print("\n" + "=" * 70)
    print("LAYER SUMMARY".center(70))
    print("=" * 70)
    
    from inference.inference import load_model
    
    try:
        model = load_model()
        
        # Group layers by type
        layer_types = {}
        for layer in model.layers:
            layer_type = type(layer).__name__
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += 1
        
        print("\nLayer Types:")
        for layer_type in sorted(layer_types.keys()):
            count = layer_types[layer_type]
            print(f"  {layer_type}: {count}")
        
        # Show sample layers
        print(f"\nFirst 10 Layers:")
        for i, layer in enumerate(model.layers[:10]):
            config = layer.get_config()
            print(f"  {i+1}. {layer.name}")
            print(f"     Type: {type(layer).__name__}")
            print(f"     Output: {layer.output_shape}")
        
        if len(model.layers) > 10:
            print(f"  ... ({len(model.layers) - 10} more layers)")
        
    except Exception as e:
        print(f"Error: {e}")


def predict_and_explain(image_path=None):
    """Run prediction and show detailed results."""
    print("\n" + "=" * 70)
    print("PREDICTION TEST".center(70))
    print("=" * 70)
    
    if image_path is None:
        # Use or create test image
        print("\nNo image provided, creating test image...")
        test_image_path = create_test_image()
        image_path = test_image_path
        print(f"Test image: {image_path}")
    else:
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return
    
    try:
        from inference.inference import predict, load_model, preprocess_image
        import numpy as np
        
        model = load_model()
        
        # Preprocess
        print(f"\nPreprocessing image...")
        x = preprocess_image(str(image_path))
        print(f"  Input array shape: {x.shape}")
        print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
        
        # Predict
        print(f"\nRunning inference...")
        result = predict(str(image_path), model=model)
        
        print(f"  Predicted class: {result['species']}")
        print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        # Show top predictions
        raw_output = np.array(result['raw'][0])
        top_k = 5
        
        import tensorflow as tf
        probs = tf.nn.softmax(raw_output).numpy()
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        from model.model_config import load_class_indices
        classes = load_class_indices()
        
        print(f"\n  Top {top_k} Predictions:")
        for rank, idx in enumerate(top_indices, 1):
            class_name = classes.get(int(idx), f"Unknown_{idx}")
            confidence = float(probs[idx])
            bar = "█" * int(confidence * 30)
            print(f"    {rank}. {class_name}: {confidence:.4f} {bar}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def create_test_image():
    """Create a simple test image."""
    from PIL import Image
    import numpy as np
    
    test_image_path = project_root / "model" / "model_test_image.png"
    
    # Create a random RGB image
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(test_image_path)
    
    return test_image_path


def show_class_mapping():
    """Display complete class mapping."""
    print("\n" + "=" * 70)
    print("CLASS MAPPING".center(70))
    print("=" * 70)
    
    from model.model_config import load_class_indices
    
    try:
        classes = load_class_indices()
        
        print(f"\nTotal Classes: {len(classes)}\n")
        
        # Separate primary and additional
        primary = {i: classes[i] for i in range(min(3, len(classes)))}
        additional = {i: classes[i] for i in range(3, len(classes))}
        
        print("Primary Bacterial Species:")
        for idx, name in primary.items():
            print(f"  Class {idx}: {name}")
        
        if additional:
            print(f"\nAdditional Classes/Variants ({len(additional)} total):")
            for idx, name in sorted(additional.items()):
                print(f"  Class {idx}: {name}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run model information display."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " AI_MICROSCOPE MODEL INFORMATION ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    print("\nOptions:")
    print("  [1] Model Information")
    print("  [2] Layer Summary")
    print("  [3] Class Mapping")
    print("  [4] Test Prediction")
    print("  [5] Full Report (all above)")
    print("  [q] Quit")
    
    try:
        choice = input("\nSelect option [1-5, q]: ").strip().lower()
        
        if choice == "1":
            show_model_info()
        elif choice == "2":
            show_model_info()
            show_layer_summary()
        elif choice == "3":
            show_class_mapping()
        elif choice == "4":
            predict_and_explain()
        elif choice == "5":
            show_model_info()
            show_layer_summary()
            show_class_mapping()
            predict_and_explain()
        elif choice == "q":
            print("Exiting...")
            return 0
        else:
            print("Invalid option")
            return 1
        
        print("\n" + "=" * 70 + "\n")
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
