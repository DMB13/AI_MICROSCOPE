"""Performance optimization and quantization utilities for AI_MICROSCOPE.

Provides tools for:
- Model quantization (fp16, int8)
- TensorFlow Lite conversion
- Inference profiling and benchmarking
- Memory optimization
- Batch processing utilities

Usage:
    python scripts/optimize_model.py
    python scripts/benchmark_inference.py
"""

import time
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def benchmark_inference(model, num_runs: int = 10) -> Dict[str, float]:
    """Benchmark model inference speed.
    
    Args:
        model: Keras model
        num_runs: Number of inference runs for averaging
        
    Returns:
        Dictionary with timing statistics
    """
    import tensorflow as tf
    
    # Create dummy input
    dummy_input = np.random.randn(1, 224, 224, 3).astype('float32')
    
    # Warmup
    _ = model.predict(dummy_input, verbose=0)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        times.append(time.perf_counter() - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }


def profile_model(model) -> Dict[str, any]:
    """Profile model for memory usage and parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = model.count_params()
    
    # Estimate model size
    # Assuming float32 (4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # Count layers
    layer_types = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    return {
        'total_parameters': total_params,
        'estimated_size_mb': model_size_mb,
        'num_layers': len(model.layers),
        'layer_types': layer_types,
    }


def convert_to_tflite(model_path: str, output_path: str = None,
                     quantization_type: str = 'float32') -> Path:
    """Convert Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to Keras model
        output_path: Path for output .tflite file
        quantization_type: 'float32', 'float16', or 'int8'
        
    Returns:
        Path to converted TFLite model
        
    Raises:
        ImportError: If TensorFlow Lite is not available
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required for TFLite conversion")
    
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization
    if quantization_type == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization_type == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For int8, we would need representative dataset
        # This is a basic implementation
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    if output_path is None:
        output_path = str(Path(model_path).with_suffix('.tflite'))
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tflite_model)
    
    return output


def compare_model_sizes(original_path: str, quantized_path: str = None) -> Dict[str, any]:
    """Compare sizes of original and quantized models.
    
    Args:
        original_path: Path to original Keras model
        quantized_path: Path to quantized model (optional)
        
    Returns:
        Dictionary with size comparison
    """
    original_size = Path(original_path).stat().st_size / (1024 * 1024)
    
    result = {
        'original_size_mb': original_size,
    }
    
    if quantized_path and Path(quantized_path).exists():
        quantized_size = Path(quantized_path).stat().st_size / (1024 * 1024)
        result['quantized_size_mb'] = quantized_size
        result['compression_ratio'] = original_size / quantized_size
        result['size_reduction_percent'] = ((original_size - quantized_size) / original_size) * 100
    
    return result


class InferenceOptimizer:
    """Optimize inference performance."""
    
    @staticmethod
    def enable_gpu_memory_growth():
        """Enable GPU memory growth to avoid OOM.
        
        Should be called before any TensorFlow operations.
        """
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def batch_predict(model, image_paths: list, batch_size: int = 8) -> list:
        """Run batch predictions on multiple images.
        
        Args:
            model: Keras model
            image_paths: List of image file paths
            batch_size: Batch size for inference
            
        Returns:
            List of predictions
        """
        from inference.inference import preprocess_image, load_class_indices
        
        predictions = []
        class_map = load_class_indices()
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                x = preprocess_image(path)
                batch_images.append(x[0])
            
            # Stack images
            batch = np.array(batch_images)
            
            # Predict
            preds = model.predict(batch, verbose=0)
            
            # Process predictions
            for pred in preds:
                if len(pred) > 1:
                    prob = tf.nn.softmax(pred).numpy()
                else:
                    prob = tf.nn.sigmoid(pred).numpy()
                
                idx = np.argmax(prob)
                confidence = float(prob[idx])
                species = class_map.get(int(idx), str(idx))
                
                predictions.append({
                    'species': species,
                    'confidence': confidence,
                    'raw': pred.tolist()
                })
        
        return predictions
    
    @staticmethod
    def compile_for_inference(model):
        """Prepare model for efficient inference.
        
        Args:
            model: Keras model
            
        Returns:
            Model (may return a different object)
        """
        try:
            import tensorflow as tf
            
            # Convert to concrete function for tracing
            run_model = tf.function(lambda x: model(x, training=False))
            concrete_func = run_model.get_concrete_function(
                tf.TensorSpec([1, 224, 224, 3], tf.float32)
            )
            
            return concrete_func
        except Exception:
            return model


def main():
    """Run optimization analysis."""
    print("\n" + "=" * 70)
    print("AI_MICROSCOPE PERFORMANCE OPTIMIZATION".center(70))
    print("=" * 70)
    
    from inference.inference import load_model
    
    try:
        print("\nLoading model...")
        model = load_model()
        
        print("\n1. PROFILING")
        print("-" * 70)
        stats = profile_model(model)
        print(f"Total parameters: {stats['total_parameters']:,}")
        print(f"Estimated size: {stats['estimated_size_mb']:.2f} MB")
        print(f"Number of layers: {stats['num_layers']}")
        print(f"Layer types: {dict(sorted(stats['layer_types'].items()))}")
        
        print("\n2. INFERENCE BENCHMARKING")
        print("-" * 70)
        print("Warming up...")
        timings = benchmark_inference(model, num_runs=5)
        print(f"Mean inference time: {timings['mean']*1000:.1f} ms")
        print(f"Std deviation: {timings['std']*1000:.1f} ms")
        print(f"Min time: {timings['min']*1000:.1f} ms")
        print(f"Max time: {timings['max']*1000:.1f} ms")
        print(f"Median time: {timings['median']*1000:.1f} ms")
        
        print("\n3. THROUGHPUT ANALYSIS")
        print("-" * 70)
        single_image_time = timings['mean']
        throughput = 1.0 / single_image_time
        print(f"Single image inference: {single_image_time*1000:.1f} ms")
        print(f"Throughput: {throughput:.1f} images/second")
        print(f"Batch 8: ~{throughput*8:.1f} images/second (estimated)")
        
        print("\n4. OPTIMIZATION RECOMMENDATIONS")
        print("-" * 70)
        
        if timings['mean'] < 0.5:
            print("✓ Inference speed is excellent (<500ms)")
        elif timings['mean'] < 1.0:
            print("⚠ Inference speed is acceptable (500-1000ms)")
        else:
            print("✗ Consider quantization or batch processing")
        
        if stats['estimated_size_mb'] < 50:
            print("✓ Model size is compact (<50MB)")
        else:
            print("⚠ Consider quantization to reduce model size")
        
        print("\n5. QUANTIZATION OPTIONS")
        print("-" * 70)
        print("Available quantization methods:")
        print("  - float32: Full precision (baseline)")
        print("  - float16: Half precision (2x compression)")
        print("  - int8: Integer quantization (4x compression)")
        print("  - TFLite: Mobile/edge deployment")
        
        print("\n" + "=" * 70)
        print("Use scripts/optimize_model.py <method> to apply optimization")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
