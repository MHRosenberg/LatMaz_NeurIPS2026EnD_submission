#!/usr/bin/env python3
"""
Report Generator Utility
Creates PDF reports containing plots, summaries, and key code snippets.
"""

import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from PIL import Image as PILImage
import textwrap


def create_report(
    title: str,
    summary: str,
    plots: list,  # List of image paths
    code_snippets: dict = None,  # {"description": "code"}
    tables: list = None,  # List of {"title": str, "data": [[row1], [row2]]}
    output_dir: str = "./reports",
    filename: str = None
):
    """
    Create a PDF report with plots, summary, and code.

    Args:
        title: Report title
        summary: Text summary/description
        plots: List of paths to plot images
        code_snippets: Dict mapping description to code string
        tables: List of table dicts with 'title' and 'data' keys
        output_dir: Output directory for the PDF
        filename: Output filename (auto-generated if None)

    Returns:
        Path to the generated PDF
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    if filename is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in title[:30])
        filename = f"{timestamp}_{safe_title}.pdf"

    if not filename.endswith('.pdf'):
        filename += '.pdf'

    output_path = os.path.join(output_dir, filename)

    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10,
        leading=14
    )
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=8,
        fontName='Courier',
        backColor=colors.Color(0.95, 0.95, 0.95),
        leftIndent=10,
        rightIndent=10,
        spaceBefore=5,
        spaceAfter=10
    )

    # Build content
    content = []

    # Title
    content.append(Paragraph(title, title_style))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    content.append(Spacer(1, 20))

    # Summary
    content.append(Paragraph("Summary", heading_style))
    # Handle newlines in summary
    for para in summary.split('\n\n'):
        if para.strip():
            content.append(Paragraph(para.replace('\n', '<br/>'), body_style))
    content.append(Spacer(1, 10))

    # Tables
    if tables:
        content.append(Paragraph("Results", heading_style))
        for tbl in tables:
            if 'title' in tbl:
                content.append(Paragraph(tbl['title'], body_style))

            if 'data' in tbl and tbl['data']:
                t = Table(tbl['data'])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.8, 0.8, 0.8)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                content.append(t)
                content.append(Spacer(1, 15))

    # Plots
    if plots:
        content.append(Paragraph("Figures", heading_style))
        for i, plot_path in enumerate(plots):
            if os.path.exists(plot_path):
                # Get image dimensions to scale properly
                try:
                    with PILImage.open(plot_path) as img:
                        img_width, img_height = img.size

                    # Scale to fit page width (max 6.5 inches)
                    max_width = 6.5 * inch
                    max_height = 8 * inch

                    scale = min(max_width / img_width, max_height / img_height)
                    display_width = img_width * scale
                    display_height = img_height * scale

                    content.append(Paragraph(f"Figure {i+1}: {os.path.basename(plot_path)}", body_style))
                    content.append(Image(plot_path, width=display_width, height=display_height))
                    content.append(Spacer(1, 15))
                except Exception as e:
                    content.append(Paragraph(f"[Error loading {plot_path}: {e}]", body_style))
            else:
                content.append(Paragraph(f"[Plot not found: {plot_path}]", body_style))

    # Code snippets
    if code_snippets:
        content.append(PageBreak())
        content.append(Paragraph("Key Code", heading_style))
        for desc, code in code_snippets.items():
            content.append(Paragraph(f"<b>{desc}</b>", body_style))
            # Truncate very long code
            code_lines = code.split('\n')
            if len(code_lines) > 50:
                code = '\n'.join(code_lines[:50]) + '\n... [truncated]'

            # Use Preformatted for code to preserve formatting
            content.append(Preformatted(code, code_style))
            content.append(Spacer(1, 10))

    # Build PDF
    doc.build(content)
    print(f"Report saved: {output_path}")

    return output_path


def quick_report(title, summary, plot_dir, output_dir="./reports"):
    """
    Quick report from a directory of plots.

    Args:
        title: Report title
        summary: Text summary
        plot_dir: Directory containing PNG plots
        output_dir: Output directory
    """
    # Find all PNG files in the directory
    plots = []
    if os.path.isdir(plot_dir):
        plots = sorted([
            os.path.join(plot_dir, f)
            for f in os.listdir(plot_dir)
            if f.endswith('.png')
        ])

    return create_report(
        title=title,
        summary=summary,
        plots=plots,
        output_dir=output_dir
    )


# Example usage
if __name__ == "__main__":
    # Test report
    test_summary = """
    This is a test report demonstrating the report generation capability.

    Key findings:
    - Point 1: Example finding
    - Point 2: Another finding
    - Point 3: Third finding
    """

    create_report(
        title="Test Report",
        summary=test_summary,
        plots=[],
        tables=[{
            'title': 'Example Table',
            'data': [
                ['Column A', 'Column B', 'Column C'],
                ['1', '2', '3'],
                ['4', '5', '6']
            ]
        }],
        output_dir="./reports",
        filename="test_report.pdf"
    )
