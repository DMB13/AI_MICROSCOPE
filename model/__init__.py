"""Model package for AI_MICROSCOPE

Provides database management, model loading, and clinical record handling.
"""

from .db import Database, get_db, close_db

__all__ = ["Database", "get_db", "close_db"]
