from pdfminer.high_level import extract_text
import tempfile

def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    text = extract_text(tmp_path)
    return text or ""