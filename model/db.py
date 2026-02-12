"""SQLite helper for AI_MICROSCOPE

Provides a small, thread-safe API for initializing the DB, inserting records,
querying recent records, and exporting to CSV.

Usage example:
    db = Database()
    db.insert_record(patient_id="P123", species="E.coli", confidence=0.95,
                     image_path="/path/to/img.jpg", gradcam_path="/path/to/gc.jpg")
    rows = db.get_recent(10)
    db.export_csv("export.csv")
    db.close()
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading
import datetime
import csv
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_DB_NAME = "clinical_records.db"
SCHEMA_FILENAME = "clinical_records_schema.sql"

class Database:
    def __init__(self, db_path: Optional[str] = None, ensure_schema: bool = True):
        self._lock = threading.RLock()
        if db_path is None:
            db_path = str(Path(__file__).resolve().parent / DEFAULT_DB_NAME)
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if ensure_schema:
            self._ensure_schema()
        logger.debug("Database initialized at %s", self.db_path)

    def _ensure_schema(self):
        schema_path = Path(__file__).resolve().parent / SCHEMA_FILENAME
        if schema_path.exists():
            ddl = schema_path.read_text(encoding="utf-8")
            with self._lock:
                cur = self.conn.cursor()
                cur.executescript(ddl)
                self.conn.commit()
                cur.close()
            logger.debug("Applied schema from %s", schema_path)
        else:
            # Fallback DDL in case schema file missing
            ddl = (
                "CREATE TABLE IF NOT EXISTS clinical_records ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "patient_id TEXT NOT NULL,"
                "timestamp TEXT NOT NULL,"
                "species TEXT,"
                "confidence REAL,"
                "image_path TEXT,"
                "gradcam_path TEXT"
                ");"
            )
            with self._lock:
                cur = self.conn.cursor()
                cur.execute(ddl)
                self.conn.commit()
                cur.close()
            logger.debug("Applied fallback schema")

    def insert_record(self, patient_id: str, species: Optional[str], confidence: Optional[float],
                      image_path: Optional[str], gradcam_path: Optional[str],
                      timestamp: Optional[str] = None) -> int:
        """Insert a clinical record and return the inserted row id."""
        if timestamp is None:
            timestamp = datetime.datetime.utcnow().isoformat()
        sql = (
            "INSERT INTO clinical_records (patient_id, timestamp, species, confidence, image_path, gradcam_path) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        params = (patient_id, timestamp, species, confidence, image_path, gradcam_path)
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(sql, params)
            rowid = cur.lastrowid
            self.conn.commit()
            cur.close()
        logger.debug("Inserted record id=%s patient=%s species=%s", rowid, patient_id, species)
        return rowid

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent records as a list of dicts ordered by newest first."""
        sql = "SELECT * FROM clinical_records ORDER BY timestamp DESC LIMIT ?"
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(sql, (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
        return rows

    def export_csv(self, csv_path: str, limit: Optional[int] = None) -> Path:
        """Export records to CSV. If `limit` is provided, export only that many recent rows."""
        records = self.get_recent(limit) if limit is not None else self.get_recent(1000000)
        out = Path(csv_path)
        fieldnames = ["id", "patient_id", "timestamp", "species", "confidence", "image_path", "gradcam_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in reversed(records):  # write oldest first
                writer.writerow({k: r.get(k) for k in fieldnames})
        logger.debug("Exported %d records to %s", len(records), out)
        return out

    def close(self):
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass
        logger.debug("Database connection closed")


# Convenience module-level functions
_db_singleton: Optional[Database] = None

def get_db(db_path: Optional[str] = None) -> Database:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = Database(db_path=db_path)
    return _db_singleton

def close_db():
    global _db_singleton
    if _db_singleton is not None:
        _db_singleton.close()
        _db_singleton = None
