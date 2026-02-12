-- SQL schema for clinical records
CREATE TABLE IF NOT EXISTS clinical_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    species TEXT,
    confidence REAL,
    image_path TEXT,
    gradcam_path TEXT
);
