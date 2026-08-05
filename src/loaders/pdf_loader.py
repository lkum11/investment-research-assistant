from pypdf import PdfReader

def load_pdf(path:str) -> str:
    """Take a PDF file path, return all its text as one string."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() +"\n"
    return text