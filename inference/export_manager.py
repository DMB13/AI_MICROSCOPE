"""Clinical report export module for AI_MICROSCOPE.

Provides comprehensive export functionality for clinical records in multiple formats:
- CSV (comma-separated values)
- JSON (structured data)
- HTML (human-readable reports)
- PDF (print-ready clinical reports)

Usage:
    from reports import export_manager
    
    exporter = export_manager.ReportExporter()
    exporter.export_csv("reports/export.csv", limit=100)
    exporter.export_json("reports/data.json", limit=100)
    exporter.export_html("reports/report.html", limit=100)
    exporter.export_pdf("reports/clinical_report.pdf", limit=100, include_images=True)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import datetime
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ClinicalRecord:
    """Represents a single clinical record."""
    id: int
    patient_id: str
    timestamp: str
    species: str
    confidence: float
    image_path: Optional[str]
    gradcam_path: Optional[str]


class ReportExporter:
    """Export clinical records in various formats."""
    
    FIELD_NAMES = [
        "id", "patient_id", "timestamp", "species", 
        "confidence", "image_path", "gradcam_path"
    ]
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize exporter.
        
        Args:
            db_path: Optional path to SQLite database. If None, uses default.
        """
        from model.db import get_db
        self.db = get_db(db_path)
    
    def export_csv(self, output_path: str, limit: Optional[int] = None) -> Path:
        """Export records to CSV format.
        
        Args:
            output_path: Path where CSV will be saved
            limit: Maximum number of records to export (None = all)
            
        Returns:
            Path to exported file
        """
        records = self.db.get_recent(limit or 1000000)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELD_NAMES)
            writer.writeheader()
            
            # Reverse to write oldest first
            for record in reversed(records):
                # Convert Row to dict if needed
                if hasattr(record, 'keys'):
                    row_dict = dict(record)
                else:
                    row_dict = asdict(record)
                
                writer.writerow({k: row_dict.get(k) for k in self.FIELD_NAMES})
        
        logger.info(f"Exported {len(records)} records to CSV: {output}")
        return output
    
    def export_json(self, output_path: str, limit: Optional[int] = None, 
                   pretty: bool = True) -> Path:
        """Export records to JSON format.
        
        Args:
            output_path: Path where JSON will be saved
            limit: Maximum number of records to export (None = all)
            pretty: Pretty-print JSON (indent 2 spaces)
            
        Returns:
            Path to exported file
        """
        records = self.db.get_recent(limit or 1000000)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        records_list = []
        for record in reversed(records):
            if hasattr(record, 'keys'):
                row_dict = dict(record)
            else:
                row_dict = asdict(record)
            records_list.append(row_dict)
        
        # Add metadata
        data = {
            "export_timestamp": datetime.datetime.utcnow().isoformat(),
            "total_records": len(records_list),
            "records": records_list
        }
        
        with output.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2 if pretty else None, ensure_ascii=False)
        
        logger.info(f"Exported {len(records)} records to JSON: {output}")
        return output
    
    def export_html(self, output_path: str, limit: Optional[int] = None,
                   title: str = "Clinical Records Report") -> Path:
        """Export records to HTML format for web viewing.
        
        Args:
            output_path: Path where HTML will be saved
            limit: Maximum number of records to export (None = all)
            title: Report title
            
        Returns:
            Path to exported file
        """
        records = self.db.get_recent(limit or 1000000)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self._generate_html(records, title)
        
        with output.open("w", encoding="utf-8") as fh:
            fh.write(html_content)
        
        logger.info(f"Exported {len(records)} records to HTML: {output}")
        return output
    
    def export_pdf(self, output_path: str, limit: Optional[int] = None,
                  title: str = "AI_MICROSCOPE Clinical Report",
                  include_images: bool = False) -> Path:
        """Export records to PDF format.
        
        Args:
            output_path: Path where PDF will be saved
            limit: Maximum number of records to export (None = all)
            title: Report title
            include_images: Include Grad-CAM images in PDF (requires reportlab)
            
        Returns:
            Path to exported file
        """
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError:
            logger.warning("pdf export requires reportlab. Install with: pip install reportlab")
            # Fallback to HTML export
            html_path = str(output_path).replace(".pdf", ".html")
            return self.export_html(html_path, limit, title)
        
        records = self.db.get_recent(limit or 1000000)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF
        doc = SimpleDocTemplate(str(output), pagesize=A4)
        story = []
        
        # Title
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        story.append(Paragraph(title, title_style))
        
        # Metadata
        timestamp = datetime.datetime.utcnow().isoformat()
        meta_style = ParagraphStyle(
            'Meta',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(f"Generated: {timestamp} | Records: {len(records)}", meta_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Table data
        table_data = [["ID", "Patient ID", "Timestamp", "Species", "Confidence"]]
        
        for record in reversed(records):
            if hasattr(record, 'keys'):
                row_dict = dict(record)
            else:
                row_dict = asdict(record)
            
            table_data.append([
                str(row_dict.get('id', '')),
                str(row_dict.get('patient_id', '')),
                str(row_dict.get('timestamp', ''))[:19],  # Truncate timestamp
                str(row_dict.get('species', '')),
                f"{float(row_dict.get('confidence', 0))*100:.1f}%"
            ])
        
        # Create table
        table = Table(table_data, colWidths=[0.7*inch, 1.2*inch, 1.5*inch, 2*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            "AI_MICROSCOPE v1.0 | Decision Support System | For Research Use Only",
            footer
        ))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Exported {len(records)} records to PDF: {output}")
        return output
    
    def _generate_html(self, records: List[Dict], title: str) -> str:
        """Generate HTML content for records.
        
        Args:
            records: List of record dictionaries
            title: Report title
            
        Returns:
            HTML string
        """
        timestamp = datetime.datetime.utcnow().isoformat()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="ai-microscope-fields" content="id,patient_id,timestamp,species,confidence,image_path,gradcam_path">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 40px;
        }}
        
        header {{
            text-align: center;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: #1f77b4;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .metadata {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-box {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border-left: 4px solid #1f77b4;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #1f77b4;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background-color: #1f77b4;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        .confidence {{
            font-weight: 600;
            color: #27ae60;
        }}
        
        .confidence.low {{
            color: #e74c3c;
        }}
        
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.85em;
        }}
        
        @media print {{
            body {{
                background-color: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="metadata">Generated: {timestamp}</div>
        </header>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(records)}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(set(r.get('patient_id') if hasattr(r, 'get') else dict(r).get('patient_id') for r in records))}</div>
                <div class="stat-label">Unique Patients</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(set(r.get('species') if hasattr(r, 'get') else dict(r).get('species') for r in records))}</div>
                <div class="stat-label">Unique Species</div>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Patient ID</th>
                    <th>Timestamp</th>
                    <th>Species</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for record in reversed(records):
            if hasattr(record, 'get'):
                row_dict = record
            else:
                row_dict = dict(record)
            
            confidence = float(row_dict.get('confidence', 0))
            confidence_class = "low" if confidence < 0.80 else ""
            
            html += f"""                <tr>
                    <td>{row_dict.get('id', '')}</td>
                    <td>{row_dict.get('patient_id', '')}</td>
                    <td>{str(row_dict.get('timestamp', ''))[:19]}</td>
                    <td>{row_dict.get('species', '')}</td>
                    <td class="confidence {confidence_class}">{confidence*100:.1f}%</td>
                </tr>
"""
        
        html += """            </tbody>
        </table>
        
        <footer>
            <p>AI_MICROSCOPE v1.0 | Clinical Decision Support System</p>
            <p>This report is for research and evaluation purposes only.</p>
            <p>Decisions should be made by qualified microbiologists in consultation with clinical staff.</p>
        </footer>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    exporter = ReportExporter()
    
    # Example exports
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    print("Exporting clinical records...")
    csv_path = exporter.export_csv(str(exports_dir / "clinical_export.csv"))
    print(f"✓ CSV: {csv_path}")
    
    json_path = exporter.export_json(str(exports_dir / "clinical_data.json"))
    print(f"✓ JSON: {json_path}")
    
    html_path = exporter.export_html(str(exports_dir / "clinical_report.html"))
    print(f"✓ HTML: {html_path}")
    
    try:
        pdf_path = exporter.export_pdf(str(exports_dir / "clinical_report.pdf"))
        print(f"✓ PDF: {pdf_path}")
    except Exception as e:
        print(f"⚠ PDF export skipped: {e}")
