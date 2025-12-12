import fitz

def extract_text_from_pdf(path):
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text

