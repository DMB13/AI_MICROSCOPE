"""Model configuration and constants for AI_MICROSCOPE.

This module centralizes all model-related configuration including:
- Model file paths and names
- Class indices mapping
- Model input/output specifications
- Preprocessing parameters
"""

from pathlib import Path
from typing import Dict
import json

# Base directories
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent

# Model file configuration
MODEL_FILENAME = "best_microscope_fusion.keras"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# Preferred model file names (in order of preference)
PREFERRED_MODEL_NAMES = [
    "best_microscope_fusion.keras",
    "best_microscope_fusion.h5",
    "best_bacteria_model.h5",
]

# Class indices configuration
CLASS_INDICES_FILE = MODEL_DIR / "class_indices.json"
DEFAULT_CLASS_INDICES = {
    0: "Escherichia_coli",
    1: "Staphylococcus_aureus",
    2: "Klebsiella_pneumoniae",
    **{i: f"Class_{i}" for i in range(3, 39)}
}

# Model specifications
MODEL_INPUT_SIZE = (224, 224)  # Expected input image size
MODEL_INPUT_SHAPE = (1, 224, 224, 3)  # Batch size 1, height, width, channels
NUM_CLASSES = 39  # Number of output classes from the model

# Preprocessing parameters
PREPROCESSING_SCALE = 255.0  # Normalize pixel values to [0, 1]
IMAGE_FORMAT = "RGB"
INTERPOLATION_METHOD = "BILINEAR"

# Record storage
RECORDS_DIR = MODEL_DIR / "records"
DATABASE_NAME = "clinical_records.db"
SCHEMA_FILENAME = "clinical_records_schema.sql"

# TFLite export path
TFLITE_DIR = MODEL_DIR / "tflite"
TFLITE_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_FILENAME = "best_microscope_fusion.tflite"
TFLITE_PATH = TFLITE_DIR / TFLITE_FILENAME


def load_class_indices() -> Dict[int, str]:
    """Load class indices from JSON file or return defaults.
    
    Returns:
        Dictionary mapping class index to class name
    """
    if CLASS_INDICES_FILE.exists():
        try:
            data = json.loads(CLASS_INDICES_FILE.read_text(encoding='utf-8'))
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            print(f"Warning: Could not load class indices from {CLASS_INDICES_FILE}: {e}")
    
    return DEFAULT_CLASS_INDICES


def find_model_file() -> Path:
    """Find the model file in MODEL_DIR.
    
    Searches for model files in this order:
    1. Exact preferred names
    2. Files starting with preferred base names
    
    Returns:
        Path to model file
        
    Raises:
        FileNotFoundError: If no model file found
    """
    # Try exact preferred names first
    for name in PREFERRED_MODEL_NAMES:
        candidate = MODEL_DIR / name
        if candidate.exists() and candidate.is_file():
            return candidate
    
    # Try files with preferred base names
    for base_name in ["best_microscope_fusion", "best_bacteria_model"]:
        for file in MODEL_DIR.iterdir():
            if file.is_file() and file.name.startswith(base_name):
                return file
    
    # Not found
    raise FileNotFoundError(
        f"No model file found in {MODEL_DIR}. "
        f"Expected one of: {PREFERRED_MODEL_NAMES}"
    )


if __name__ == "__main__":
    # Print configuration summary
    print("AI_MICROSCOPE Model Configuration")
    print("=" * 50)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Model file: {MODEL_PATH}")
    print(f"Model exists: {MODEL_PATH.exists()}")
    
    try:
        found_model = find_model_file()
        print(f"Found model: {found_model}")
        print(f"Model size: {found_model.stat().st_size / 1024 / 1024:.1f} MB")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    print(f"\nClass indices: {load_class_indices()}")
    print(f"Model input size: {MODEL_INPUT_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
