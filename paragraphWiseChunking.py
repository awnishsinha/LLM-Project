from llama_index.core.schema import TextNode
from pypdf import PdfReader
import re

def load_pdf_paragraph_wise(pdf_path):
    reader = PdfReader(pdf_path)
    nodes = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Split by blank lines
        paragraphs = re.split(r"\n\s*\n", text)

        para_no = 1

        for para in paragraphs:
            para = para.strip()

            if not para:
                continue

            node = TextNode(
                text=para,
                metadata={
                    "file_name": pdf_path,
                    "page_no": page_no,
                    "paragraph_no": para_no,
                    "chunk_type": "PARAGRAPH"
                }
            )

            nodes.append(node)
            para_no += 1

    return nodes

