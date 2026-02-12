"""Small script to create the DB and insert sample records.
Run from project root:

    python scripts/sample_insert.py

"""
import pathlib
from model.db import Database, get_db, close_db

def main():
    db_path = pathlib.Path(__file__).resolve().parents[1] / "model" / "clinical_records.db"
    db = Database(db_path=str(db_path))

    # Insert sample records
    db.insert_record(patient_id="P0001", species="Escherichia_coli", confidence=0.92,
                     image_path="model/sample_1.jpg", gradcam_path="model/sample_1_gc.jpg")
    db.insert_record(patient_id="P0002", species="Staphylococcus_aureus", confidence=0.87,
                     image_path="model/sample_2.jpg", gradcam_path="model/sample_2_gc.jpg")

    rows = db.get_recent(10)
    print("Recent records:")
    for r in rows:
        print(r)

    # Export CSV to project root
    export_path = pathlib.Path(__file__).resolve().parents[1] / "exports" / "clinical_export.csv"
    db.export_csv(str(export_path), limit=50)
    print(f"Exported CSV to {export_path}")

    db.close()

if __name__ == "__main__":
    main()
