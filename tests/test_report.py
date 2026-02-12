import tempfile
from pathlib import Path
from PIL import Image

from model.db import Database
from model.report import export_records_pdf


def test_export_records_pdf_creates_file():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # create sample images
        img = td / 'sample.png'
        gc = td / 'sample_gc.png'
        Image.new('RGB', (100, 100), color=(255, 0, 0)).save(img)
        Image.new('RGB', (100, 100), color=(0, 255, 0)).save(gc)

        # create temporary database
        db_path = td / 'test.db'
        db = Database(db_path=str(db_path))
        db.insert_record(patient_id='TEST', species='Escherichia_coli', confidence=0.95, image_path=str(img), gradcam_path=str(gc))

        records = db.get_recent(10)
        out_pdf = td / 'out.pdf'
        p = export_records_pdf(records, str(out_pdf))
        assert Path(p).exists()
        assert out_pdf.stat().st_size > 0
