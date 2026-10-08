"""
docs/build_pdf.py - Ubah laporan Markdown menjadi PDF
======================================================
Markdown → HTML (dengan gaya cetak A4) → PDF lewat Microsoft Edge headless.

Pemakaian:
    python docs/build_pdf.py                                   # laporan perbandingan model
    python docs/build_pdf.py docs/LAPORAN_SESI_01.md           # laporan lain
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @bottom-right { content: "Halaman " counter(page) " / " counter(pages); font: 8pt "Segoe UI", sans-serif; color: #666; }
  @bottom-left { content: "Laporan Eksperimen CIFAKE Detection"; font: 8pt "Segoe UI", sans-serif; color: #666; }
}
html { font: 10pt/1.5 "Segoe UI", Calibri, Arial, sans-serif; color: #1a1a1a; }
body { margin: 0; }
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 10pt; color: #0b2545; }
h2 { font-size: 14pt; margin: 20pt 0 8pt; padding-bottom: 3pt; border-bottom: 1.5pt solid #0b2545; color: #0b2545;
     break-after: avoid; }
h2.page-start { break-before: page; margin-top: 0; }
h3 { font-size: 11.5pt; margin: 14pt 0 6pt; color: #13315c; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
hr { border: 0; border-top: 0.75pt solid #ccc; margin: 12pt 0; }
a { color: #13315c; text-decoration: none; }
strong { color: #000; }
code { font: 8.5pt Consolas, monospace; background: #f1f3f5; padding: 0 3pt; border-radius: 2pt; }
pre { background: #f1f3f5; padding: 8pt 10pt; border-radius: 3pt; overflow: hidden; break-inside: avoid; }
pre code { background: none; padding: 0; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.5pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #0b2545; color: #fff; font-weight: 600; text-align: left; padding: 4pt 6pt; }
td { padding: 3.5pt 6pt; border-bottom: 0.5pt solid #d0d4da; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f6f7f9; }
img { max-width: 100%; display: block; margin: 6pt auto; break-inside: avoid; }
p:has(> img:only-child) { break-inside: avoid; margin: 8pt 0; }
td img { margin: 0 auto; }
table:has(img) th { text-align: center; }
table:has(img) tbody tr td { background: #fff; }
ul, ol { padding-left: 18pt; }
"""


def build(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    # Halaman 1 = judul + daftar isi; isi laporan dimulai di halaman baru (h2 kedua)
    first = body.find("<h2")
    second = body.find("<h2", first + 1)
    if second != -1:
        body = body[:second] + '<h2 class="page-start"' + body[second + 3:]
    title = text.splitlines()[0].lstrip("# ").strip()
    base = md_path.parent.resolve().as_uri() + "/"
    html = (f'<!doctype html><html lang="id"><head><meta charset="utf-8"><base href="{base}">'
            f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")

    pdf_path = md_path.with_suffix(".pdf").resolve()
    edge = next((p for p in EDGE_PATHS if os.path.exists(p)), None)
    if edge is None:
        sys.exit("[ERROR] Microsoft Edge tidak ditemukan.")

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "laporan.html"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run([
            edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            f"--user-data-dir={Path(tmp) / 'profile'}",
            f"--print-to-pdf={pdf_path}", html_path.as_uri(),
        ], check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[PDF] {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS / "LAPORAN_PERBANDINGAN_MODEL.md"
    build(target)
