from pathlib import Path
from pypdf import PdfReader

def load_pdfs(folder="../data/raw"):
    pages = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        reader = PdfReader(pdf)
        n_ok = 0
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"doc": pdf.name, "page": i, "text": text})
                n_ok += 1
        print(f"{pdf.name}: {n_ok}/{len(reader.pages)} pages with text")
    return pages