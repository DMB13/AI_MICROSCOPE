"""Comprehensive unit and integration tests for AI_MICROSCOPE.

Test suite covering:
- Image preprocessing
- Model inference
- Grad-CAM visualization
- Database operations
- Export functionality
- API contracts

Run tests with:
    pytest tests/test_suite.py -v
    pytest tests/test_suite.py::TestPreprocessing -v
    pytest tests/test_suite.py -k "inference" -v
"""

import sys
import pytest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import sqlite3

# Add parent to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestPreprocessing:
    """Test image preprocessing pipeline."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            yield f.name
        
        Path(f.name).unlink()
    
    def test_preprocess_image_shape(self, sample_image):
        """Test that preprocessing returns correct shape."""
        from inference.inference import preprocess_image
        from model.model_config import MODEL_INPUT_SIZE
        
        x = preprocess_image(sample_image, target_size=MODEL_INPUT_SIZE)
        
        assert x.shape == (1, 224, 224, 3), f"Expected shape (1, 224, 224, 3), got {x.shape}"
    
    def test_preprocess_image_normalization(self, sample_image):
        """Test that image is normalized to [0, 1]."""
        from inference.inference import preprocess_image
        from model.model_config import MODEL_INPUT_SIZE
        
        x = preprocess_image(sample_image, target_size=MODEL_INPUT_SIZE)
        
        assert x.min() >= 0.0 and x.max() <= 1.0, \
            f"Expected values in [0, 1], got [{x.min()}, {x.max()}]"
    
    def test_preprocess_image_rgb_conversion(self, sample_image):
        """Test that image is converted to RGB."""
        from inference.inference import preprocess_image
        from model.model_config import MODEL_INPUT_SIZE
        
        x = preprocess_image(sample_image, target_size=MODEL_INPUT_SIZE)
        
        # Should have 3 channels (RGB)
        assert x.shape[-1] == 3, f"Expected 3 channels, got {x.shape[-1]}"
    
    def test_preprocess_invalid_file(self):
        """Test that invalid file raises error."""
        from inference.inference import preprocess_image
        
        with pytest.raises(FileNotFoundError):
            preprocess_image("nonexistent_file.jpg")
    
    def test_preprocess_custom_size(self, sample_image):
        """Test preprocessing with custom target size."""
        from inference.inference import preprocess_image
        
        custom_size = (256, 256)
        x = preprocess_image(sample_image, target_size=custom_size)
        
        assert x.shape == (1, 256, 256, 3), f"Expected shape (1, 256, 256, 3), got {x.shape}"


class TestModelLoading:
    """Test model loading and caching."""
    
    def test_model_loads_successfully(self):
        """Test that model loads without error."""
        from inference.inference import load_model
        
        model = load_model()
        assert model is not None, "Model should not be None"
    
    def test_model_caching(self):
        """Test that model is cached after first load."""
        from inference.inference import load_model, _MODEL
        
        model1 = load_model()
        model2 = load_model()
        
        # Should return same cached instance
        assert model1 is model2, "Model should be cached"
    
    def test_model_input_shape(self):
        """Test that model has expected input shape."""
        from inference.inference import load_model
        
        model = load_model()
        
        # Model should accept (batch, 224, 224, 3)
        assert model.input_shape == (None, 224, 224, 3), \
            f"Expected input shape (None, 224, 224, 3), got {model.input_shape}"
    
    def test_model_output_shape(self):
        """Test that model has expected output shape."""
        from inference.inference import load_model
        
        model = load_model()
        
        # Model should output 39 classes
        assert model.output_shape[-1] == 39, \
            f"Expected 39 output classes, got {model.output_shape[-1]}"


class TestInference:
    """Test model inference pipeline."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            yield f.name
        
        Path(f.name).unlink()
    
    def test_predict_returns_dict(self, sample_image):
        """Test that predict returns expected dictionary."""
        from inference.inference import predict
        
        result = predict(sample_image)
        
        assert isinstance(result, dict), "predict should return a dict"
        assert "species" in result, "Result should contain 'species'"
        assert "confidence" in result, "Result should contain 'confidence'"
        assert "raw" in result, "Result should contain 'raw'"
    
    def test_predict_confidence_range(self, sample_image):
        """Test that confidence is in valid range [0, 1]."""
        from inference.inference import predict
        
        result = predict(sample_image)
        confidence = result["confidence"]
        
        assert 0.0 <= confidence <= 1.0, \
            f"Confidence should be in [0, 1], got {confidence}"
    
    def test_predict_species_is_string(self, sample_image):
        """Test that species is a string."""
        from inference.inference import predict
        
        result = predict(sample_image)
        
        assert isinstance(result["species"], str), "Species should be a string"
        assert len(result["species"]) > 0, "Species should not be empty"
    
    def test_predict_raw_output_shape(self, sample_image):
        """Test that raw output has correct shape."""
        from inference.inference import predict
        
        result = predict(sample_image)
        raw = result["raw"]
        
        assert len(raw) == 1, "Raw output should have batch size 1"
        assert len(raw[0]) == 39, "Raw output should have 39 classes"
    
    def test_predict_with_preloaded_model(self, sample_image):
        """Test inference with pre-loaded model."""
        from inference.inference import predict, load_model
        
        model = load_model()
        result = predict(sample_image, model=model)
        
        assert isinstance(result, dict), "predict should return a dict"


class TestGradCAM:
    """Test Grad-CAM visualization."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            yield f.name
        
        Path(f.name).unlink()
    
    def test_grad_cam_returns_image(self, sample_image):
        """Test that grad_cam returns PIL Image."""
        from inference.inference import grad_cam
        from PIL import Image as PILImage
        
        result = grad_cam(sample_image)
        
        assert isinstance(result, PILImage.Image), "grad_cam should return PIL Image"
    
    def test_grad_cam_output_size(self, sample_image):
        """Test that grad_cam output has expected size."""
        from inference.inference import grad_cam
        
        result = grad_cam(sample_image, upsample_size=(512, 512))
        
        assert result.size == (512, 512), f"Expected size (512, 512), got {result.size}"
    
    def test_grad_cam_rgb_format(self, sample_image):
        """Test that grad_cam returns RGB image."""
        from inference.inference import grad_cam
        
        result = grad_cam(sample_image)
        
        assert result.mode == 'RGB', f"Expected RGB mode, got {result.mode}"


class TestDatabase:
    """Test database operations."""
    
    @pytest.fixture
    def test_db(self):
        """Create temporary test database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        from model.db import Database
        db = Database(db_path=db_path)
        
        yield db
        
        db.close()
        Path(db_path).unlink()
    
    def test_insert_record(self, test_db):
        """Test inserting a record."""
        record_id = test_db.insert_record(
            patient_id="TEST001",
            species="Escherichia_coli",
            confidence=0.95,
            image_path="test.jpg",
            gradcam_path="test_gc.jpg"
        )
        
        assert isinstance(record_id, int), "insert_record should return int"
        assert record_id > 0, "Record ID should be positive"
    
    def test_get_recent(self, test_db):
        """Test retrieving recent records."""
        test_db.insert_record("P001", "Escherichia_coli", 0.95, "img1.jpg", None)
        test_db.insert_record("P002", "Staphylococcus_aureus", 0.87, "img2.jpg", None)
        
        records = test_db.get_recent(10)
        
        assert len(records) == 2, "Should retrieve 2 records"
        assert records[0]["patient_id"] in ["P001", "P002"]
    
    def test_get_recent_limit(self, test_db):
        """Test get_recent with limit."""
        for i in range(10):
            test_db.insert_record(f"P{i:03d}", "Unknown", 0.5, f"img{i}.jpg", None)
        
        records = test_db.get_recent(5)
        
        assert len(records) == 5, "Should respect limit"
    
    def test_records_have_standard_fields(self, test_db):
        """Test that records have expected fields."""
        test_db.insert_record("P001", "Escherichia_coli", 0.95, "img.jpg", "gc.jpg")
        records = test_db.get_recent(1)
        
        required_fields = ["id", "patient_id", "timestamp", "species", "confidence", "image_path", "gradcam_path"]
        for field in required_fields:
            assert field in records[0], f"Record should have {field} field"


class TestExports:
    """Test report export functionality."""
    
    @pytest.fixture
    def test_db_with_records(self):
        """Create test database with sample records."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        from model.db import Database
        db = Database(db_path=db_path)
        
        # Insert sample records
        for i in range(5):
            db.insert_record(
                patient_id=f"P{i:03d}",
                species=["Escherichia_coli", "Staphylococcus_aureus", "Klebsiella_pneumoniae"][i % 3],
                confidence=0.80 + (i * 0.02),
                image_path=f"img{i}.jpg",
                gradcam_path=f"gc{i}.jpg"
            )
        
        yield db
        
        db.close()
        Path(db_path).unlink()
    
    def test_export_csv(self, test_db_with_records):
        """Test CSV export."""
        from inference.export_manager import ReportExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ReportExporter(test_db_with_records.db_path)
            csv_path = exporter.export_csv(str(Path(tmpdir) / "test.csv"))
            
            assert csv_path.exists(), "CSV file should be created"
            assert csv_path.stat().st_size > 0, "CSV file should not be empty"
            
            # Verify CSV structure
            with open(csv_path) as f:
                content = f.read()
                assert "patient_id" in content, "CSV should have header"
                assert "P000" in content, "CSV should have data"
    
    def test_export_json(self, test_db_with_records):
        """Test JSON export."""
        from inference.export_manager import ReportExporter
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ReportExporter(test_db_with_records.db_path)
            json_path = exporter.export_json(str(Path(tmpdir) / "test.json"))
            
            assert json_path.exists(), "JSON file should be created"
            
            # Verify JSON structure
            with open(json_path) as f:
                data = json.load(f)
                assert "records" in data, "JSON should have records"
                assert "export_timestamp" in data, "JSON should have timestamp"
                assert len(data["records"]) > 0, "JSON should have data"
    
    def test_export_html(self, test_db_with_records):
        """Test HTML export."""
        from inference.export_manager import ReportExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ReportExporter(test_db_with_records.db_path)
            html_path = exporter.export_html(str(Path(tmpdir) / "test.html"))
            
            assert html_path.exists(), "HTML file should be created"
            
            # Verify HTML structure
            with open(html_path) as f:
                content = f.read()
                assert "<html" in content.lower(), "Should be valid HTML"
                assert "patient_id" in content.lower(), "Should have patient data"


class TestClassIndices:
    """Test class index loading."""
    
    def test_load_class_indices(self):
        """Test loading class indices."""
        from inference.inference import load_class_indices
        
        classes = load_class_indices()
        
        assert isinstance(classes, dict), "Should return dict"
        assert len(classes) == 39, "Should have 39 classes"
    
    def test_class_indices_mapping(self):
        """Test that class indices map to species names."""
        from inference.inference import load_class_indices
        
        classes = load_class_indices()
        
        # Check primary classes
        assert classes[0] == "Escherichia_coli"
        assert classes[1] == "Staphylococcus_aureus"
        assert classes[2] == "Klebsiella_pneumoniae"


class TestAPIContracts:
    """Test API contracts and interfaces."""
    
    def test_inference_module_exports(self):
        """Test that inference module exports required functions."""
        from inference import inference
        
        required_functions = ["load_model", "predict", "grad_cam", "preprocess_image"]
        for func in required_functions:
            assert hasattr(inference, func), f"inference module should export {func}"
    
    def test_database_exports(self):
        """Test that database module exports required classes."""
        from model.db import Database, get_db, close_db
        
        assert Database is not None
        assert callable(get_db)
        assert callable(close_db)
    
    def test_export_manager_exports(self):
        """Test that export_manager exports required classes."""
        from inference.export_manager import ReportExporter
        
        assert ReportExporter is not None
        
        # Test instantiation
        exporter = ReportExporter()
        assert hasattr(exporter, 'export_csv')
        assert hasattr(exporter, 'export_json')
        assert hasattr(exporter, 'export_html')


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
