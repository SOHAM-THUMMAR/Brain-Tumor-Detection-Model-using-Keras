from docx.shared import Pt, Inches
from docx.enum.table import WD_TABLE_ALIGNMENT
from docs.styles import (
    add_styled_heading, add_body_p, set_cell_background, set_cell_margins, format_run, set_section_header_footer
)


def build_chapters_4_to_6(doc, ch_sec):
    # CHAPTER 4.0 SYSTEM ANALYSIS
    set_section_header_footer(ch_sec, header_title="4.0 System Analysis")
    add_styled_heading(doc, "4.0 SYSTEM ANALYSIS", level=1)

    add_styled_heading(doc, "4.1 Analysis of Existing System", level=2)
    add_body_p(
        doc,
        "Existing diagnosis relies on qualitative radiologist visual inspection. Key flaws include: (1) High diagnostic latency during emergency triaging, (2) Subjective boundary interpretation leading to missed low-contrast tumors, and (3) Lack of automated explainability visualization."
    )

    add_styled_heading(doc, "4.2 Proposed Automated CNN System", level=2)
    add_body_p(
        doc,
        "The proposed system integrates automated image resizing (224x224x3), 1/255 rescaling, CNN feature extraction, and Grad-CAM visual heatmaps inside a Flask application, achieving immediate objective feedback."
    )

    add_styled_heading(doc, "4.3 Dataset Modeling & Distribution", level=2)
    add_body_p(doc, "The dataset contains brain MRI slices organized into Training, Validation, and Testing sets as detailed in Table 4.1:")

    t_ds = doc.add_table(rows=1, cols=4)
    t_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    ds_hdr = t_ds.rows[0].cells
    ds_hdr[0].text = "Dataset Split"
    ds_hdr[1].text = "Tumor Scans"
    ds_hdr[2].text = "No Tumor Scans"
    ds_hdr[3].text = "Total Images"
    for cell in ds_hdr:
        set_cell_background(cell, "003366")
        for p in cell.paragraphs:
            for r in p.runs:
                format_run(r, size_pt=10, bold=True, color_rgb=(255, 255, 255))

    ds_data = [
        ("Training Set", "4,500", "2,000", "6,500"),
        ("Validation Set", "800", "400", "1,200"),
        ("Testing Set", "458", "208", "666"),
        ("Total Project Dataset", "5,758", "2,608", "8,366"),
    ]
    for d_split, d_tum, d_not, d_tot in ds_data:
        r_c = t_ds.add_row().cells
        r_c[0].text = d_split
        r_c[1].text = d_tum
        r_c[2].text = d_not
        r_c[3].text = d_tot
        for i, cell in enumerate(r_c):
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            for p in cell.paragraphs:
                for r in p.runs:
                    format_run(r, size_pt=10, bold=(i == 0))

    add_body_p(doc, "Table 4.1: Dataset Distribution across Train, Validation, and Test Sets")

    doc.add_page_break()

    # CHAPTER 5.0 SYSTEM DESIGN
    set_section_header_footer(ch_sec, header_title="5.0 System Design")
    add_styled_heading(doc, "5.0 SYSTEM DESIGN", level=1)

    add_styled_heading(doc, "5.1 System Architecture", level=2)
    add_body_p(
        doc,
        "The system follows a three-tier architecture: (1) Presentation Tier: HTML5/CSS3/Vanilla JS browser interface, (2) Application Tier: Python Flask web application server with routes (/predict, /stats, /health), and (3) Deep Learning Engine: TensorFlow/Keras CNN model singleton and Grad-CAM module."
    )

    add_styled_heading(doc, "5.2 CNN Model Layer Design", level=2)
    add_body_p(
        doc,
        "The custom CNN architecture consists of the following sequential layers:\n"
        "• Conv2D (16 filters, 3x3 kernel, ReLU activation, input shape: 224x224x3)\n"
        "• Conv2D (32 filters, 3x3 kernel, ReLU activation)\n"
        "• MaxPool2D (2x2 pool size)\n"
        "• Conv2D (64 filters, 3x3 kernel, ReLU activation)\n"
        "• MaxPool2D (2x2 pool size)\n"
        "• Conv2D (128 filters, 3x3 kernel, ReLU activation)\n"
        "• MaxPool2D (2x2 pool size)\n"
        "• Dropout (0.25 rate)\n"
        "• Flatten Layer\n"
        "• Dense (64 units, ReLU activation)\n"
        "• Dropout (0.25 rate)\n"
        "• Dense (1 unit, Sigmoid activation head)"
    )

    doc.add_page_break()

    # CHAPTER 6.0 IMPLEMENTATION PLANNING
    set_section_header_footer(ch_sec, header_title="6.0 Implementation Planning")
    add_styled_heading(doc, "6.0 IMPLEMENTATION PLANNING", level=1)

    add_styled_heading(doc, "6.1 Module Specification", level=2)
    add_body_p(doc, "• app.py: Flask entrypoint containing HTTP routing, file upload handling, and error views.")
    add_body_p(doc, "• model_utils.py: Singleton model loader, image preprocessor, threshold inference, and Grad-CAM overlay generator.")
    add_body_p(doc, "• config.py: Configuration constants (paths, image dimension 224x224, file size limits, threshold 0.3).")

    add_styled_heading(doc, "6.2 Sample Core Implementation Code", level=2)
    add_body_p(doc, "The code snippet below illustrates the model inference and recall-optimized threshold logic from app/services/model_service.py:")

    p_code = doc.add_paragraph()
    p_code.paragraph_format.left_indent = Inches(0.4)
    p_code.paragraph_format.space_before = Pt(6)
    p_code.paragraph_format.space_after = Pt(12)
    r_c = p_code.add_run(
        "def predict(model, processed_image):\n"
        "    \"\"\"\n"
        "    Runs model inference and applies classification threshold (0.3).\n"
        "    Returns (label: str, confidence: float 0-100).\n"
        "    \"\"\"\n"
        "    raw_pred = model.predict(processed_image, verbose=0)\n"
        "    prob = float(raw_pred[0][0])\n\n"
        "    if prob >= Config.CLASSIFICATION_THRESHOLD:  # 0.3 threshold\n"
        "        label = 'Tumor'\n"
        "        confidence = prob * 100.0\n"
        "    else:\n"
        "        label = 'No Tumor'\n"
        "        confidence = (1.0 - prob) * 100.0\n\n"
        "    return label, round(confidence, 2)\n"
    )
    format_run(r_c, font_name="Courier New", size_pt=9.5, color_rgb=(0, 51, 102))

    doc.add_page_break()
