"""Inference utilities for AI_MICROSCOPE.

Provides:
- `load_model` — load a Keras model from disk (lazy cached)
- `predict` — run preprocessing + model prediction and map to species name
- `grad_cam` — compute Grad-CAM heatmap overlay as a PIL.Image

This module imports TensorFlow lazily and raises informative errors if TF
is not available in the runtime. The functions accept either a loaded
Keras model or a path to the model file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np
import json

# Import model configuration
from model.model_config import (
    MODEL_DIR,
    MODEL_PATH,
    PREFERRED_MODEL_NAMES,
    CLASS_INDICES_FILE,
    MODEL_INPUT_SIZE,
    find_model_file,
    load_class_indices as load_class_indices_from_config,
)

# cached model and class mapping
_MODEL = None
_CLASS_INDICES = None


def _ensure_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is required for model inference and Grad-CAM. "
            "Install it (see requirements) or run without model support." ) from e


def load_class_indices(path: Optional[Path] = None) -> Dict[int, str]:
    """Load class indices from JSON file.
    
    Args:
        path: Optional path to class indices JSON. If not provided, uses configured path.
        
    Returns:
        Dictionary mapping class index to class name
    """
    global _CLASS_INDICES
    if _CLASS_INDICES is not None:
        return _CLASS_INDICES
    
    # Use provided path or load from config
    if path is None:
        _CLASS_INDICES = load_class_indices_from_config()
    else:
        if not Path(path).exists():
            raise FileNotFoundError(f"class indices JSON not found: {path}")
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        _CLASS_INDICES = {int(k): v for k, v in data.items()}
    
    return _CLASS_INDICES


def load_model(model_path: Optional[str] = None):
    """Load and return a Keras model. Caches the model in-module.

    Arguments:
        model_path: path to the saved Keras model file. If omitted,
            falls back to auto-detection in MODEL_DIR using configured
            preferred names.
            
    Returns:
        Loaded Keras model (not compiled)
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    
    tf = _ensure_tf()
    
    if model_path is None:
        # Use configured model discovery
        try:
            found_path = find_model_file()
            model_path = str(found_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No model file found in {MODEL_DIR}. {str(e)}") from e
    else:
        model_path = str(model_path)
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model without compiling to speed up load and avoid custom compile requirements
    _MODEL = tf.keras.models.load_model(model_path, compile=False)
    return _MODEL


def preprocess_image(path: str, target_size: tuple = MODEL_INPUT_SIZE) -> np.ndarray:
    """Preprocess image for model input.
    
    Args:
        path: Path to image file
        target_size: Target image size (height, width). Defaults to MODEL_INPUT_SIZE.
        
    Returns:
        Preprocessed image as numpy array with shape (1, height, width, 3)
    """
    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        target_size = MODEL_INPUT_SIZE
    
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def _find_last_conv_layer(model, tf):
    # heuristic: find layer with 4D output and 'conv' in name, start from end
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            out_shape = None
        if out_shape is None:
            continue
        if isinstance(out_shape, tuple) and len(out_shape) == 4:
            if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                return layer.name
    # fallback: return last layer with 4D output
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            out_shape = None
        if out_shape and isinstance(out_shape, tuple) and len(out_shape) == 4:
            return layer.name
    raise RuntimeError('Could not find a conv layer for Grad-CAM')


def predict(image_path: str, model=None) -> Dict[str, Any]:
    """Run model prediction on `image_path` and return dict with keys:
    - `species`: predicted class name (str)
    - `confidence`: probability of predicted class (float)
    - `raw`: full model output (list)
    
    Args:
        image_path: Path to input image
        model: Optional pre-loaded model. If None, loads model using load_model().
        
    Returns:
        Dictionary containing prediction results
    """
    tf = _ensure_tf()
    if model is None:
        model = load_model()
    # accept either model object or path
    if isinstance(model, (str, Path)):
        model = load_model(str(model))

    x = preprocess_image(image_path, target_size=MODEL_INPUT_SIZE)
    preds = model.predict(x)
    # handle models that output logits or probabilities
    probs = tf.nn.softmax(preds[0]).numpy() if preds.shape[-1] > 1 else tf.nn.sigmoid(preds[0]).numpy()
    class_map = load_class_indices()
    idx = int(np.argmax(probs))
    species = class_map.get(idx, str(idx))
    confidence = float(probs[idx])
    return {"species": species, "confidence": confidence, "raw": preds.tolist()}


def grad_cam(image_path: str, model=None, upsample_size=(512, 512)) -> Optional[Image.Image]:
    """Compute Grad-CAM overlay image for the top predicted class.

    Args:
        image_path: Path to input image
        model: Optional pre-loaded model. If None, loads model using load_model().
        upsample_size: Size to upsample the heatmap to. Defaults to (512, 512).

    Returns:
        PIL.Image (RGB) with the heatmap overlayed on the original image,
        or None if Grad-CAM computation fails.
        
    Raises:
        RuntimeError: If TensorFlow is not available
    """
    tf = _ensure_tf()
    if model is None:
        model = load_model()
    if isinstance(model, (str, Path)):
        model = load_model(str(model))

    img = Image.open(image_path).convert('RGB')
    orig_size = img.size
    x = preprocess_image(image_path, target_size=MODEL_INPUT_SIZE)

    # predict and find top class
    preds = model.predict(x)
    if preds.ndim == 2 and preds.shape[1] > 1:
        top_index = int(tf.argmax(preds[0]).numpy())
    else:
        top_index = 0

    try:
        last_conv_name = _find_last_conv_layer(model, tf)

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tf.convert_to_tensor(x))
            if predictions.shape[-1] > 1:
                loss = predictions[:, top_index]
            else:
                loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        # compute guided gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap.numpy(), 0)
        max_val = heatmap.max() if heatmap.max() != 0 else 1e-10
        heatmap = heatmap / max_val

        # resize heatmap to original image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize(upsample_size, resample=Image.BILINEAR).convert('L')

    except RuntimeError:
        # Fallback for models without convolutional layers: use input-gradient saliency
        x_tensor = tf.convert_to_tensor(x)
        x_tensor = tf.cast(x_tensor, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = model(x_tensor)
            if predictions.shape[-1] > 1:
                loss = predictions[:, top_index]
            else:
                loss = predictions[:, 0]
        grads = tape.gradient(loss, x_tensor)[0]
        # aggregate gradients across channels to form a saliency map
        saliency = tf.reduce_mean(tf.math.abs(grads), axis=-1).numpy()
        saliency = np.maximum(saliency, 0)
        max_val = saliency.max() if saliency.max() != 0 else 1e-10
        saliency = saliency / max_val
        heatmap = np.uint8(255 * saliency)
        heatmap = Image.fromarray(heatmap).resize(upsample_size, resample=Image.BILINEAR).convert('L')

    # overlay (shared for both conv-based and saliency fallback)
    # create a solid semi-transparent red overlay image of the same size
    heatmap_color = Image.new('RGBA', heatmap.size, (255, 0, 0, 120))
    orig = img.resize(upsample_size, Image.BILINEAR).convert('RGBA')
    heatmap_mask = heatmap.convert('L')
    overlay = Image.composite(heatmap_color, orig, heatmap_mask)
    final = Image.alpha_composite(orig, overlay).convert('RGB')
    return final
