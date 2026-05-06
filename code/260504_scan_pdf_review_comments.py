#!/usr/bin/env python3
"""
260504_scan_pdf_review_comments.py

Scan PDF(s) for review-style annotations (Highlight, FreeText, Text, Note,
Comment, StrikeOut, Underline, Squiggly, Caret) and print a table with the
underlying body text where applicable.

Workflow (per Author 2026-05-04):
  1. cld snapshots a draft as paper/c<ts>_main_DandB_draft_<topic>.pdf.
  2. Author opens it, annotates, saves under paper/c<ts>_main_DandB_draft_<topic>.pdf
     (cld and Author initials swapped; Author's prefix moved to the left).
  3. cld runs this script with default args to find the most recently
     mtime'd file matching the MR_ pattern and print its annotations
     + the body text under each highlight.

Usage:
  python 260504_scan_pdf_review_comments.py
      # → scans the most recent paper/c*_*.pdf

  python 260504_scan_pdf_review_comments.py --glob 'paper/c*.pdf'
      # → scans all matching files

  python 260504_scan_pdf_review_comments.py --file <path>
      # → scans a single file
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


PROJECT_ROOT = Path('<REPO_ROOT>')

REVIEW_SUBTYPES = {
    '/Highlight', '/FreeText', '/Text', '/Note', '/Comment',
    '/StrikeOut', '/Underline', '/Squiggly', '/Caret',
}

DEFAULT_GLOB = 'paper/c*_*.pdf'


def find_most_recent(pattern: str) -> Path | None:
    matches = sorted(
        glob.glob(str(PROJECT_ROOT / pattern)),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return Path(matches[0]) if matches else None


def text_under_rect(pdf_path: Path, page: int, rect: tuple[float, float, float, float],
                    page_height: float = 792.0, pad: float = 4.0) -> str:
    """Use pdftotext -bbox-layout to extract words whose bounding box centre
    falls inside the given PDF rect (origin bottom-left, points). Returns
    the joined word string, or '' if pdftotext is unavailable.
    """
    try:
        out = subprocess.run(
            ['pdftotext', '-bbox-layout', '-f', str(page), '-l', str(page),
             str(pdf_path), '-'],
            capture_output=True, text=True, errors='replace', check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ''

    # PDF rect is bottom-left origin; pdftotext uses top-left.
    x_min_pdf, y_min_pdf, x_max_pdf, y_max_pdf = rect
    y_top = page_height - y_max_pdf
    y_bot = page_height - y_min_pdf

    words = re.findall(
        r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)"[^>]*>([^<]+)</word>',
        out,
    )
    found = []
    for xMin, yMin, xMax, yMax, w in words:
        xMin, yMin, xMax, yMax = map(float, (xMin, yMin, xMax, yMax))
        cx = (xMin + xMax) / 2
        cy = (yMin + yMax) / 2
        if (y_top - pad) <= cy <= (y_bot + pad) and (x_min_pdf - pad) <= cx <= (x_max_pdf + pad):
            found.append((cy, xMin, w))
    found.sort()
    return ' '.join(w for _, _, w in found).strip()


def scan_file(path: Path) -> list[dict]:
    r = PdfReader(str(path))
    annots = []
    for i, page in enumerate(r.pages):
        if '/Annots' not in page:
            continue
        # Letter is 612 x 792 by default; respect the page's MediaBox.
        try:
            mb = page.get('/MediaBox')
            page_h = float(mb[3]) if mb else 792.0
        except Exception:
            page_h = 792.0
        for a_ref in page['/Annots']:
            a = a_ref.get_object()
            sub = str(a.get('/Subtype', ''))
            if sub not in REVIEW_SUBTYPES:
                continue
            rect = a.get('/Rect', [0, 0, 0, 0])
            try:
                rect_t = tuple(float(v) for v in rect)
            except Exception:
                rect_t = (0.0, 0.0, 0.0, 0.0)
            annots.append({
                'file': path.name,
                'page': i + 1,
                'subtype': sub,
                'author': str(a.get('/T', '')),
                'contents': str(a.get('/Contents', '')),
                'rect': rect_t,
                'underlying_text': text_under_rect(path, i + 1, rect_t, page_h),
            })
    return annots


def print_table(rows: list[dict]) -> None:
    if not rows:
        print('No review annotations found.')
        return
    for r in rows:
        print(f"[{r['file']}  p{r['page']}]  {r['subtype']}  by {r['author'] or '<no name>'}")
        if r['contents']:
            print(f"  comment:    {r['contents']}")
        if r['underlying_text']:
            txt = r['underlying_text']
            if len(txt) > 200:
                txt = txt[:200] + ' …'
            print(f"  underlying: {txt}")
        print(f"  rect: {r['rect']}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default=DEFAULT_GLOB,
                    help=f'glob (relative to PROJECT_ROOT) of files to scan; default={DEFAULT_GLOB}')
    ap.add_argument('--file', help='specific PDF to scan (overrides --glob)')
    ap.add_argument('--all', action='store_true',
                    help='scan all files matching --glob (default: only most recent)')
    args = ap.parse_args()

    if args.file:
        targets = [Path(args.file)]
    elif args.all:
        targets = [Path(p) for p in sorted(
            glob.glob(str(PROJECT_ROOT / args.glob)),
            key=lambda p: os.path.getmtime(p), reverse=True,
        )]
    else:
        latest = find_most_recent(args.glob)
        if latest is None:
            print(f'No files matching {args.glob} found under {PROJECT_ROOT}', file=sys.stderr)
            sys.exit(1)
        targets = [latest]

    for t in targets:
        rows = scan_file(t)
        if len(targets) > 1 and not rows:
            continue
        if len(targets) > 1:
            print(f'=== {t.relative_to(PROJECT_ROOT)} ===')
        print_table(rows)


if __name__ == '__main__':
    main()
