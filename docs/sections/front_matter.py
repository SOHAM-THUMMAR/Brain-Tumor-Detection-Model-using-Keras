from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docs.config import (
    STUDENT_NAME, ENROLLMENT_NO, PROJECT_TITLE, FULL_TITLE,
    DEPARTMENT, GUIDE_NAME, INSTITUTE_NAME, UNIVERSITY_NAME, ACADEMIC_YEAR
)
from docs.styles import format_run, add_styled_heading, add_body_p


def build_cover_page(doc):
    p_cov_title = doc.add_paragraph()
    p_cov_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cov_title.paragraph_format.space_before = Pt(36)
    r = p_cov_title.add_run("A PROJECT REPORT ON\n")
    format_run(r, size_pt=14, bold=True)
    r = p_cov_title.add_run(f"{FULL_TITLE}\n")
    format_run(r, size_pt=18, bold=True, color_rgb=(0, 51, 102))

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(24)
    p_sub.paragraph_format.space_after = Pt(36)
    r = p_sub.add_run("Submitted in partial fulfillment of the requirements for the degree of\n")
    format_run(r, size_pt=12, italic=True)
    r = p_sub.add_run("BACHELOR OF TECHNOLOGY\n")
    format_run(r, size_pt=14, bold=True)
    r = p_sub.add_run(f"in\n{DEPARTMENT}\n")
    format_run(r, size_pt=12, bold=True)

    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_before = Pt(36)
    p_by.paragraph_format.space_after = Pt(36)
    r = p_by.add_run("Submitted By:\n")
    format_run(r, size_pt=12, bold=True)
    r = p_by.add_run(f"{STUDENT_NAME} [Enrollment No: {ENROLLMENT_NO}]\n\n")
    format_run(r, size_pt=12, bold=True)

    r = p_by.add_run("Under the Guidance of:\n")
    format_run(r, size_pt=12, bold=True)
    r = p_by.add_run(f"{GUIDE_NAME}\n{DEPARTMENT}\n\n")
    format_run(r, size_pt=12)

    r = p_by.add_run(f"{INSTITUTE_NAME}\n{UNIVERSITY_NAME}\nAcademic Year: {ACADEMIC_YEAR}\n")
    format_run(r, size_pt=12, bold=True)

    doc.add_page_break()


def build_title_page(doc):
    p_t = doc.add_paragraph()
    p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t.paragraph_format.space_before = Pt(48)
    r = p_t.add_run(f"{PROJECT_TITLE}\n\n")
    format_run(r, size_pt=16, bold=True)
    r = p_t.add_run("7th Semester B.Tech Project Report\n")
    format_run(r, size_pt=14, italic=True)

    add_body_p(doc, f"Project Title: {FULL_TITLE}")
    add_body_p(doc, f"Author / Candidate Name: {STUDENT_NAME}")
    add_body_p(doc, f"Enrollment Number: {ENROLLMENT_NO}")
    add_body_p(doc, f"Branch / Department: {DEPARTMENT}")
    add_body_p(doc, f"Internal Guide: {GUIDE_NAME}")
    add_body_p(doc, f"Institution: {INSTITUTE_NAME}")
    add_body_p(doc, "Date of Submission: [FILL SUBMISSION DATE, e.g. May 2026]")

    doc.add_page_break()


def build_declaration(doc):
    add_styled_heading(doc, "CANDIDATE'S DECLARATION", level=1)
    add_body_p(
        doc,
        f"I hereby declare that the project work entitled \"{PROJECT_TITLE}\" submitted to the {DEPARTMENT}, {INSTITUTE_NAME}, is a record of original work done by me under the guidance of {GUIDE_NAME}. This report has not been submitted elsewhere for the award of any degree, diploma, or title."
    )
    add_body_p(
        doc,
        "I further declare that all source codes, deep learning architectures, web application modules, and experimental results presented in this report are authentic and developed specifically for this project."
    )

    p_sig = doc.add_paragraph()
    p_sig.paragraph_format.space_before = Pt(48)
    r = p_sig.add_run(f"Date: [FILL DATE]\nPlace: [FILL CITY]\n\n\n_______________________\n{STUDENT_NAME}\n(Enrollment No: {ENROLLMENT_NO})")
    format_run(r, size_pt=12, bold=True)

    doc.add_page_break()


def build_certificate(doc):
    add_styled_heading(doc, "CERTIFICATE", level=1)
    add_body_p(
        doc,
        f"This is to certify that the project report entitled \"{PROJECT_TITLE}\" submitted by {STUDENT_NAME} (Enrollment No: {ENROLLMENT_NO}) has been successfully completed under my supervision in partial fulfillment of the requirements for the degree of Bachelor of Technology in {DEPARTMENT} from {UNIVERSITY_NAME} during the academic year {ACADEMIC_YEAR}."
    )

    p_cert_sig = doc.add_paragraph()
    p_cert_sig.paragraph_format.space_before = Pt(64)
    r = p_cert_sig.add_run(f"_______________________\t\t\t_______________________\n{GUIDE_NAME}\t\t\tHead of Department\n(Project Guide)\t\t\t({DEPARTMENT})\n\n\n_______________________\nExternal Examiner")
    format_run(r, size_pt=12, bold=True)

    doc.add_page_break()


def build_acknowledgement(doc):
    add_styled_heading(doc, "ACKNOWLEDGEMENT", level=1)
    add_body_p(
        doc,
        f"I express my deepest sense of gratitude and sincere thanks to my project guide, {GUIDE_NAME}, for invaluable guidance, encouragement, and insightful suggestions throughout the development of this project. His/Her technical expertise and continuous mentorship were instrumental in shaping the methodology and successful execution of the model and web application."
    )
    add_body_p(
        doc,
        f"I am also thankful to the Head of Department, Prof. [FILL HOD NAME], and all faculty members of the {DEPARTMENT} for providing the required computational infrastructure, software tools, and academic environment."
    )
    add_body_p(
        doc,
        "Finally, I express my sincere appreciation to my family and friends for their constant moral support, patience, and encouragement during the course of this major project."
    )

    doc.add_page_break()


def build_abstract(doc):
    add_styled_heading(doc, "ABSTRACT", level=1)
    add_body_p(
        doc,
        "Brain tumors represent one of the most critical and life-threatening neurological conditions. Early detection and precise localization of brain neoplasms from Magnetic Resonance Imaging (MRI) scans are vital for effective surgical planning and treatment outcomes. Manual evaluation of brain MRI slices by radiologists is time-consuming and subject to inter-observer variability. In this work, an end-to-end automated computer-aided diagnosis system is developed using custom Deep Convolutional Neural Networks (CNN) implemented in TensorFlow/Keras, integrated with a clinical-ready Python Flask web interface."
    )
    add_body_p(
        doc,
        "The proposed deep CNN model consists of four sequential convolutional blocks with ReLU activations, max-pooling, dropout regularization, and a sigmoid classification head optimized via binary crossentropy loss. To eliminate false-negative diagnosis in screening scenarios, class-weighted training and decision threshold tuning (0.3 threshold) were employed. On a comprehensive test dataset of 666 MRI slices (458 tumor, 208 non-tumor), the model achieved 98% overall accuracy, 100% tumor recall (zero missed tumor cases), and an Area Under ROC Curve (AUC) of 1.000."
    )
    add_body_p(
        doc,
        "To make the model interpretable and accessible for clinical research, Gradient-weighted Class Activation Mapping (Grad-CAM) was integrated to dynamically highlight spatial heatmaps of tumor regions. The complete pipeline was deployed into a responsive Flask web application supporting drag-and-drop file uploads, real-time inference, and performance metric visualization."
    )

    doc.add_page_break()
