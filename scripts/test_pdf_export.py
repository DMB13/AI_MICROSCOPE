"""Quick smoke-test: export recent records to PDF using model.report utilities."""
from pathlib import Path
from model.report import export_recent_pdf

if __name__ == '__main__':
    out = Path('exports') / 'clinical_export_smoke.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    p = export_recent_pdf(out_path=str(out), limit=5)
    print('PDF exported to:', p)
