import os
import sys
import docx

# Add parent directory to path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from docs.config import OUTPUT_DOCX
from docs.styles import setup_section_margins, set_section_header_footer
from docs.sections.front_matter import (
    build_cover_page, build_title_page, build_declaration,
    build_certificate, build_acknowledgement, build_abstract
)
from docs.sections.abbr_toc import build_abbreviations, build_toc_and_lists
from docs.sections.chapters_1_to_3 import build_chapters_1_to_3
from docs.sections.chapters_4_to_6 import build_chapters_4_to_6
from docs.sections.chapters_7_to_10 import build_chapters_7_to_10
from docs.sections.references import build_references_and_appendices


def main():
    print("Generating 7th Semester Project Report (.docx)...")

    doc = docx.Document()

    # Front Matter Section (Roman numeral header/footer setup)
    front_sec = doc.sections[0]
    setup_section_margins(front_sec)
    set_section_header_footer(front_sec, header_title="Front Matter", is_front_matter=True)

    build_cover_page(doc)
    build_title_page(doc)
    build_declaration(doc)
    build_certificate(doc)
    build_acknowledgement(doc)
    build_abstract(doc)

    build_abbreviations(doc)
    build_toc_and_lists(doc)

    # Main Chapters Section (Arabic numeral page numbering)
    ch_sec = doc.add_section()
    setup_section_margins(ch_sec)

    build_chapters_1_to_3(doc, ch_sec)
    build_chapters_4_to_6(doc, ch_sec)
    build_chapters_7_to_10(doc, ch_sec)

    build_references_and_appendices(doc, ch_sec)

    os.makedirs(os.path.dirname(OUTPUT_DOCX), exist_ok=True)
    doc.save(OUTPUT_DOCX)
    print(f"Successfully generated project report: {OUTPUT_DOCX}")



if __name__ == "__main__":
    main()
