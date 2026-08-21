import os
import datetime
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from app.config import Config

logger = logging.getLogger(__name__)


def generate_patient_pdf_report(
    saved_filename,
    prediction,
    confidence,
    original_abs_path,
    heatmap_abs_path,
    highlight_abs_path,
):
    """
    Generates a professional PDF Patient Diagnostic Report using ReportLab.
    Saves the PDF to Config.REPORTS_FOLDER and returns relative path ('reports/pdf_name.pdf').
    """
    try:
        os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
        pdf_filename = f"report_{saved_filename}.pdf"
        pdf_abs_path = os.path.join(Config.REPORTS_FOLDER, pdf_filename)

        doc = SimpleDocTemplate(
            pdf_abs_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom Paragraph Styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f172a"),
            alignment=1,  # Centered
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            alignment=1,
        )

        badge_tumor_style = ParagraphStyle(
            "BadgeTumor",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#ef4444"),
            alignment=1,
        )

        badge_notumor_style = ParagraphStyle(
            "BadgeNoTumor",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#10b981"),
            alignment=1,
        )

        meta_label_style = ParagraphStyle(
            "MetaLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
        )

        meta_val_style = ParagraphStyle(
            "MetaVal",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
        )

        img_caption_style = ParagraphStyle(
            "ImgCaption",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            alignment=1,
        )

        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748b"),
            alignment=1,
        )

        story = []

        # 1. Header Banner
        story.append(Paragraph("🧠 NeuroScan AI Diagnostic Report", title_style))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                "Automated Brain MRI Scan Tumor Classification & Explainability System",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 10))
        story.append(
            HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12)
        )

        # 2. Metadata Table
        report_id = f"REP-{saved_filename[:8].upper()}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        meta_data = [
            [
                Paragraph("Report ID:", meta_label_style),
                Paragraph(report_id, meta_val_style),
                Paragraph("Scan Filename:", meta_label_style),
                Paragraph(saved_filename, meta_val_style),
            ],
            [
                Paragraph("Analysis Date:", meta_label_style),
                Paragraph(now_str, meta_val_style),
                Paragraph("Model Engine:", meta_label_style),
                Paragraph("CNN Binary Classifier (Keras 3)", meta_val_style),
            ],
        ]

        meta_table = Table(meta_data, colWidths=[1.1 * inch, 2.4 * inch, 1.2 * inch, 2.5 * inch])
        meta_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(meta_table)
        story.append(Spacer(1, 14))

        # 3. Diagnostic Result Card
        if prediction == "Tumor":
            badge_text = "⚠️ DIAGNOSIS: TUMOR DETECTED"
            badge_p = Paragraph(badge_text, badge_tumor_style)
            bg_color = colors.HexColor("#fef2f2")
            border_color = colors.HexColor("#ef4444")
        else:
            badge_text = "✅ DIAGNOSIS: NO TUMOR DETECTED"
            badge_p = Paragraph(badge_text, badge_notumor_style)
            bg_color = colors.HexColor("#ecfdf5")
            border_color = colors.HexColor("#10b981")

        conf_p = Paragraph(
            f"<b>Prediction Confidence:</b> {confidence}% (Recall-Tuned Safety Threshold: 0.3)",
            ParagraphStyle("Conf", parent=styles["Normal"], fontName="Helvetica", fontSize=11, alignment=1, textColor=colors.HexColor("#1e293b")),
        )

        res_data = [[badge_p], [Spacer(1, 4)], [conf_p]]
        res_table = Table(res_data, colWidths=[7.2 * inch])
        res_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("BOX", (0, 0), (-1, -1), 1.5, border_color),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(res_table)
        story.append(Spacer(1, 16))

        # 4. Images Grid Table (Original, Highlight, Heatmap)
        img_cells = []
        caption_cells = []

        if original_abs_path and os.path.exists(original_abs_path):
            img_orig = RLImage(original_abs_path, width=2.2 * inch, height=2.2 * inch)
            img_cells.append(img_orig)
            caption_cells.append(Paragraph("1. Original MRI Scan", img_caption_style))

        if highlight_abs_path and os.path.exists(highlight_abs_path):
            img_high = RLImage(highlight_abs_path, width=2.2 * inch, height=2.2 * inch)
            img_cells.append(img_high)
            caption_cells.append(Paragraph("2. Tumor Region Highlight", img_caption_style))

        if heatmap_abs_path and os.path.exists(heatmap_abs_path):
            img_heat = RLImage(heatmap_abs_path, width=2.2 * inch, height=2.2 * inch)
            img_cells.append(img_heat)
            caption_cells.append(Paragraph("3. Grad-CAM Heatmap", img_caption_style))

        if img_cells:
            img_table_data = [img_cells, caption_cells]
            col_w = [2.35 * inch] * len(img_cells)
            img_table = Table(img_table_data, colWidths=col_w)
            img_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(img_table)
            story.append(Spacer(1, 14))

        # 5. Benchmark Performance Summary
        bench_data = [
            [
                Paragraph("<b>Test Accuracy:</b> 98%", meta_val_style),
                Paragraph("<b>Tumor Recall:</b> 100%", meta_val_style),
                Paragraph("<b>False Negatives:</b> 0", meta_val_style),
                Paragraph("<b>AUC Score:</b> 1.000", meta_val_style),
            ]
        ]
        bench_table = Table(bench_data, colWidths=[1.8 * inch] * 4)
        bench_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(bench_table)
        story.append(Spacer(1, 16))

        # 6. Disclaimer Footer
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8)
        )
        story.append(
            Paragraph(
                "<b>Medical Disclaimer:</b> This report is generated automatically by a deep learning computer-aided diagnosis system for research and decision-support purposes only. It does not replace professional radiological evaluation or clinical diagnostic procedures.",
                disclaimer_style,
            )
        )

        doc.build(story)
        logger.info(f"PDF Diagnostic Report successfully generated at {pdf_abs_path}")
        return f"reports/{pdf_filename}"

    except Exception as e:
        logger.error(f"Failed to generate PDF diagnostic report: {str(e)}", exc_info=True)
        return None
