from docx.shared import Inches, Pt
from docs.styles import add_styled_heading, add_body_p, format_run, set_section_header_footer


def build_references_and_appendices(doc, ch_sec):
    set_section_header_footer(ch_sec, header_title="References")
    add_styled_heading(doc, "REFERENCES", level=1)

    refs = [
        "[1] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. IEEE International Conference on Computer Vision (ICCV), 618-626.",
        "[2] Abadi, M., et al. (2016). TensorFlow: A System for Large-Scale Machine Learning. 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 265-283.",
        "[3] Chollet, F. (2015). Keras: Deep Learning for Humans. GitHub Repository: https://github.com/fchollet/keras",
        "[4] Grinberg, M. (2018). Flask Web Development: Developing Web Applications with Python. O'Reilly Media.",
        "[5] Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., ... & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis, 42, 60-88.",
        "[6] ReportLab Software. (2024). ReportLab PDF Generation User Guide. ReportLab Inc., https://www.reportlab.com/",
    ]
    for ref in refs:
        add_body_p(doc, ref)

    doc.add_page_break()

    set_section_header_footer(ch_sec, header_title="Appendices")
    add_styled_heading(doc, "APPENDICES", level=1)
    add_styled_heading(doc, "Appendix A: Project Directory Tree", level=2)

    tree_str = (
        "Brain-Tumor-Detection-Model-using-Keras/\n"
        "├── start.py                   # One-click application launcher (auto-venv & pip)\n"
        "├── app/\n"
        "│   ├── __init__.py            # Flask app factory (create_app)\n"
        "│   ├── app.py                 # Application entrypoint\n"
        "│   ├── config.py              # Configuration settings\n"
        "│   ├── model_utils.py         # Backward-compatible facade\n"
        "│   ├── routes/\n"
        "│   │   ├── main_routes.py\n"
        "│   │   ├── predict_routes.py  # Prediction & PDF download endpoints\n"
        "│   │   ├── stats_routes.py\n"
        "│   │   └── health_routes.py\n"
        "│   ├── services/\n"
        "│   │   ├── image_service.py\n"
        "│   │   ├── model_service.py\n"
        "│   │   ├── gradcam_service.py # Grad-CAM & bounding box contour highlight\n"
        "│   │   ├── pdf_service.py     # ReportLab PDF patient report generator\n"
        "│   │   ├── validation_service.py\n"
        "│   │   └── stats_service.py\n"
        "│   ├── templates/\n"
        "│   └── static/\n"
        "│       ├── css/\n"
        "│       ├── js/\n"
        "│       ├── uploads/\n"
        "│       ├── heatmaps/\n"
        "│       └── reports/           # Generated PDF patient reports\n"
        "├── docs/\n"
        "│   ├── config.py\n"
        "│   ├── styles.py\n"
        "│   ├── generate_report.py\n"
        "│   ├── sections/\n"
        "│   └── project_report.docx\n"
        "├── graphs/\n"
        "├── bestModel.keras\n"
        "├── requirements.txt\n"
        "└── Procfile\n"
    )
    p_tree = doc.add_paragraph()
    p_tree.paragraph_format.left_indent = Inches(0.3)
    r_tr = p_tree.add_run(tree_str)
    format_run(r_tr, font_name="Courier New", size_pt=9.5, color_rgb=(0, 51, 102))
