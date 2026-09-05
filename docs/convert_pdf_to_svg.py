import os
import fitz  # PyMuPDF

docs_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(docs_dir, "view_diagrams.pdf")

def convert_pdf_pages_to_svg():
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} does not exist.")
        return

    doc = fitz.open(pdf_path)
    print(f"Opened PDF: {pdf_path} (Total Pages: {len(doc)})")

    page_names = [
        "pdf_dfd_level0.svg",
        "pdf_dfd_level1.svg",
        "pdf_dfd_level2.svg",
        "pdf_er_diagram.svg",
        "pdf_class_architecture.svg",
        "pdf_object_interaction.svg",
        "pdf_control_flow.svg",
        "pdf_sequence_diagram.svg"
    ]

    for i, page in enumerate(doc):
        svg_content = page.get_svg_image()
        name = page_names[i] if i < len(page_names) else f"pdf_diagram_page_{i+1}.svg"
        output_svg_path = os.path.join(docs_dir, name)
        
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        
        print(f"Extracted Page {i+1} -> SVG: {output_svg_path}")

if __name__ == "__main__":
    convert_pdf_pages_to_svg()
