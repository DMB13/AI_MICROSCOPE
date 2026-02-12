"""Report generation utilities for AI_MICROSCOPE.

Provides CSV and PDF export helpers that can be used by the GUI
or CLI to export recent clinical records with embedded images.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from PIL import Image
import io
import datetime

from model.db import get_db


def _make_thumbnail_bytes(path: Optional[str], size=(120, 120)) -> Optional[io.BytesIO]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        im = Image.open(p)
        im.thumbnail(size)
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        bio.seek(0)
        return bio
    except Exception:
        return None


def export_records_pdf(records: List[Dict[str, Any]], out_path: str, title: str = "Clinical Records Export") -> Path:
    """Export given records to a nicely formatted PDF.

    Args:
        records: list of record dicts
        out_path: output PDF path
        title: document title

    Returns:
        Path to generated PDF
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out), pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=20*mm, bottomMargin=15*mm
    )
    styles = getSampleStyleSheet()
    # custom styles
    title_style = ParagraphStyle(
        name="TitleBold", parent=styles["Title"], fontSize=18, leading=22, alignment=1
    )
    meta_style = ParagraphStyle(name="Meta", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    header_style = ParagraphStyle(name="Header", parent=styles["Normal"], fontSize=10, leading=12, alignment=1, textColor=colors.whitesmoke)

    story = []

    # Header
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.datetime.utcnow().isoformat()} UTC", meta_style))
    story.append(Paragraph(f"Total records: {len(records)}", meta_style))
    story.append(Spacer(1, 12))

    # table header with dark background
    table_data = [[
        Paragraph("ID", header_style),
        Paragraph("Patient ID", header_style),
        Paragraph("Timestamp", header_style),
        Paragraph("Species", header_style),
        Paragraph("Confidence", header_style),
        Paragraph("Image", header_style),
        Paragraph("Grad-CAM", header_style)
    ]]

    for r in records:
        img_thumb = _make_thumbnail_bytes(r.get("image_path"))
        gc_thumb = _make_thumbnail_bytes(r.get("gradcam_path"))

        img_cell = RLImage(img_thumb, width=30*mm, height=30*mm) if img_thumb is not None else Paragraph("—", styles["Normal"])
        gc_cell = RLImage(gc_thumb, width=30*mm, height=30*mm) if gc_thumb is not None else Paragraph("—", styles["Normal"])

        confidence = r.get("confidence")
        conf_str = f"{confidence:.3f}" if isinstance(confidence, float) else (str(confidence) if confidence is not None else "")

        row = [
            Paragraph(str(r.get("id", "")), styles["Normal"]),
            Paragraph(r.get("patient_id", ""), styles["Normal"]),
            Paragraph(r.get("timestamp", ""), styles["Normal"]),
            Paragraph(r.get("species", ""), styles["Normal"]),
            Paragraph(conf_str, styles["Normal"]),
            img_cell,
            gc_cell
        ]
        table_data.append(row)

    col_widths = [16*mm, 30*mm, 46*mm, 46*mm, 18*mm, 36*mm, 36*mm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4b6b9a")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (4,1), (4,-1), "RIGHT"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))

    story.append(table)
    doc.build(story)
    return out


def export_recent_pdf(db=None, out_path: str = "exports/clinical_export.pdf", limit: int = 100) -> Path:
    if db is None:
        db = get_db()
    records = db.get_recent(limit)
    return export_records_pdf(records, out_path)
