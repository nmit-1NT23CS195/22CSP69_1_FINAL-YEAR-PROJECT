import fitz  # PyMuPDF
import docx
import io


def extract_text(file_bytes: bytes, filename: str = "") -> str:
    text = ""
    filename = filename.lower()

    try:
        if filename.endswith(".docx") or filename.endswith(".doc"):
            # Handle Microsoft Word
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        elif filename.endswith(".txt"):
            # Handle raw text files
            text = file_bytes.decode("utf-8", errors="ignore")
            
        else:
            # Default to PDF (PyMuPDF)
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()

    except Exception as e:
        print(f"File extraction error for {filename}:", e)
        return ""

    return text