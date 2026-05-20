from pypdf import PdfReader
from llama_index.core import Document


def load_pdf_page_wise(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": pdf_path,
                    "page_no": page_no,
                    "chunk_type": "PAGE"
                }
            )
        )

    return documents