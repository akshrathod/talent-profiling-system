from pathlib import Path
import docx

def extract_text(file_path: Path) -> str:
    doc = docx.Document(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)