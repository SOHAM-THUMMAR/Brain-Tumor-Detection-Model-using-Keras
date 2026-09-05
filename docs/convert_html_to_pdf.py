import os
import subprocess
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

docs_dir = os.path.dirname(os.path.abspath(__file__))
pdf_output_path = os.path.join(docs_dir, "view_diagrams.pdf")

def convert_via_edge():
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    ]
    html_uri = f"file:///{os.path.join(docs_dir, 'view_diagrams.html').replace(os.sep, '/')}"
    
    for exe in edge_paths:
        if os.path.exists(exe):
            cmd = [
                exe,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=10000",
                f"--print-to-pdf={pdf_output_path}",
                html_uri
            ]
            print(f"Running headless PDF conversion using: {exe}")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(pdf_output_path) and os.path.getsize(pdf_output_path) > 0:
                print(f"Successfully generated Edge PDF: {pdf_output_path}")
                return True
    return False

def generate_reportlab_pdf():
    print("Generating fallback/high-res PDF via ReportLab...")
    doc = SimpleDocTemplate(
        pdf_output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#0f172a"), alignment=1
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor("#64748b"), alignment=1
    )
    fig_title_style = ParagraphStyle(
        "FigTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#0284c7")
    )
    desc_style = ParagraphStyle(
        "DescStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#334155")
    )

    story = [
        Paragraph("🧠 Brain Tumor Detection System", title_style),
        Spacer(1, 4),
        Paragraph("Complete Project Architecture & System Diagrams Document", subtitle_style),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=15),
    ]

    diagrams = [
        ("Figure 4.1 (a): Data Flow Diagram (DFD Level 0 — Context Diagram)", "Context-level boundary mapping user inputs and system output deliverables.", "dfd_level0.png", 6.8, 3.5),
        ("Figure 4.1 (b): Data Flow Diagram (DFD Level 1 — Core Pipeline & Data Stores)", "Decomposition of system into 6 core functional processes and 5 persistent data stores.", "dfd_level1.png", 7.0, 4.1),
        ("Entity-Relationship (E-R) Diagram", "Entity-relationship model depicting USER, MRI_SCAN, MODEL_PREDICTION, VISUAL_EXPLAINABILITY, and PATIENT_PDF_REPORT.", "er_diagram.png", 7.0, 4.0),
        ("Figure 4.2: High-Level System Class Architecture Diagram", "UML Class Architecture depicting Config, Application Factory, Controller Blueprints, and Service Layer Singletons.", "class_architecture.png", 7.0, 4.5),
        ("Section 4.7.5: Object Interaction Diagram", "Dynamic method calls between instantiated runtime service objects during inference.", "object_interaction_diagram.png", 7.0, 3.8),
        ("Component Interaction & Execution Sequence Diagram", "Synchronous HTTP lifecycle from request upload guard through Grad-CAM to PDF report download.", "sequence_diagram.png", 7.0, 3.8),
    ]

    for title, desc, img_file, w_in, h_in in diagrams:
        img_path = os.path.join(docs_dir, img_file)
        story.append(Paragraph(title, fig_title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(desc, desc_style))
        story.append(Spacer(1, 8))
        if os.path.exists(img_path):
            story.append(RLImage(img_path, width=w_in * inch, height=h_in * inch))
        story.append(Spacer(1, 15))
        story.append(PageBreak())

    doc.build(story)
    print(f"Successfully generated PDF: {pdf_output_path}")

if __name__ == "__main__":
    if not convert_via_edge():
        generate_reportlab_pdf()
