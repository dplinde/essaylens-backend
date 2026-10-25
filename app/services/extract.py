"""Text extraction for digital essay files (.txt, .pdf, .docx)."""
import io

from docx import Document
from pypdf import PdfReader


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return data.decode(enc).strip()
            except UnicodeDecodeError:
                continue
    elif name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    raise UnsupportedFile("Supported formats: .txt, .pdf, .docx")
