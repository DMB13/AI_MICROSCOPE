"""Integration test and diagnostic script for AI_MICROSCOPE.

This script validates that all components are properly configured and can work together.
Run from project root:

    python scripts/integration_test.py
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL (Pillow)")
    except ImportError as e:
        print(f"✗ PIL: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        from inference import inference
        print("✓ inference module")
    except ImportError as e:
        print(f"✗ inference module: {e}")
        return False
    
    try:
        from model.db import Database, get_db, close_db
        print("✓ model.db module")
    except ImportError as e:
        print(f"✗ model.db module: {e}")
        return False
    
    try:
        from model.model_config import (
            MODEL_DIR, MODEL_PATH, find_model_file, load_class_indices
        )
        print("✓ model.model_config module")
    except ImportError as e:
        print(f"✗ model.model_config module: {e}")
        return False
    
    return True


def test_model_configuration():
    """Test that model configuration is correct."""
    print("\n" + "=" * 60)
    print("TESTING MODEL CONFIGURATION")
    print("=" * 60)
    
    from model.model_config import (
        MODEL_DIR, MODEL_PATH, find_model_file, load_class_indices,
        MODEL_INPUT_SIZE, NUM_CLASSES
    )
    
    print(f"Model directory: {MODEL_DIR}")
    print(f"Model exists: {MODEL_PATH.exists()}")
    
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"Model size: {size_mb:.1f} MB")
    
    try:
        model_file = find_model_file()
        print(f"✓ Found model file: {model_file.name}")
    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        return False
    
    try:
        class_map = load_class_indices()
        print(f"✓ Class indices loaded: {class_map}")
    except Exception as e:
        print(f"✗ Failed to load class indices: {e}")
        return False
    
    if len(class_map) != NUM_CLASSES:
        print(f"⚠ Warning: Expected {NUM_CLASSES} classes, got {len(class_map)}")
    
    print(f"Model input size: {MODEL_INPUT_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    return True


def test_model_loading():
    """Test that the Keras model can be loaded."""
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    from inference.inference import load_model
    
    try:
        model = load_model()
        print("✓ Model loaded successfully")
        
        print(f"Model type: {type(model)}")
        if hasattr(model, 'input_shape'):
            print(f"Input shape: {model.input_shape}")
        if hasattr(model, 'output_shape'):
            print(f"Output shape: {model.output_shape}")
        if hasattr(model, 'layers'):
            print(f"Number of layers: {len(model.layers)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test that the database can be initialized."""
    print("\n" + "=" * 60)
    print("TESTING DATABASE")
    print("=" * 60)
    
    try:
        from model.db import Database, get_db
        
        db = get_db()
        print("✓ Database initialized")
        
        # Test inserting a record
        record_id = db.insert_record(
            patient_id="TEST001",
            species="Escherichia_coli",
            confidence=0.95,
            image_path="test_image.jpg",
            gradcam_path="test_gradcam.jpg"
        )
        print(f"✓ Test record inserted (ID: {record_id})")
        
        # Test querying records
        records = db.get_recent(1)
        if records:
            print(f"✓ Retrieved {len(records)} record(s)")
            print(f"  Latest record: {records[0]}")
        
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_image():
    """Create a simple test image."""
    from PIL import Image
    import numpy as np
    
    test_image_path = project_root / "model" / "test_image.png"
    
    # Create a simple RGB image
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(test_image_path)
    
    return test_image_path


def test_inference():
    """Test that inference works on a test image."""
    print("\n" + "=" * 60)
    print("TESTING INFERENCE")
    print("=" * 60)
    
    try:
        test_image_path = create_test_image()
        print(f"Created test image: {test_image_path}")
        
        from inference.inference import predict, load_model
        
        model = load_model()
        result = predict(str(test_image_path), model=model)
        
        print(f"✓ Prediction successful")
        print(f"  Species: {result['species']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Raw output shape: {len(result['raw'][0])} classes")
        
        # Clean up
        test_image_path.unlink()
        
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + " AI_MICROSCOPE INTEGRATION TEST ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    tests = [
        ("Imports", test_imports),
        ("Model Configuration", test_model_configuration),
        ("Model Loading", test_model_loading),
        ("Database", test_database),
        ("Inference", test_inference),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All integration tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
