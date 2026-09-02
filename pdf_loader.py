"""
PDF loading module for the AI PDF Knowledge Assistant.
 
Responsible ONLY for opening a PDF file and extracting its raw text.
This module knows nothing about chunking, embeddings, vector storage,
or Gemini — it just answers one question: "what text is in this PDF?"
"""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

def load_pdf(file_path: str) -> str:
    """
    Extract and return all readable text from a PDF file.
 
    Args:
        file_path: Path to the PDF file on disk.
 
    Returns:
        A single string containing the text of every page, in order,
        separated by blank lines. Pages with no extractable text are
        skipped (with a warning) rather than causing a crash.
 
    Raises:
        FileNotFoundError: If the given path does not exist.
        ValueError: If the file is not a PDF, or if it can't be opened
            as a valid PDF (corrupted, encrypted without a password, etc.).
    """

    path = Path(file_path)

    # --- Problem 1: file does not exist ---
    if not path.exists():
        raise FileNotFoundError(f"Path is not a file: {file_path}")
    
    # --- Problem 2: file is not a PDF ---
    # A quick, cheap check based on the extension. This won't catch every
    # mislabeled file, but it filters out obvious mistakes (e.g. a .txt
    # file renamed by accident) before we even try to parse it.

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF (expected .pdf, got '{path.suffix}'): {file_path}")
    
    # --- Problem 3: PDF cannot be opened or read ---
    try:
        reader = PdfReader(str(path))

    except (PdfReadError, OSError) as error:
        raise ValueError(f"Could not open '{file_path}' as a valid PDF: {error}") from error
    
    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be read without a password: {file_path}")
    
    # --- Extract text page by page ---
    page_texts: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()

        except Exception as error:
            # A single malformed page shouldn't take down the whole load.
            print(f"Warning: could not extract text from page {page_number}: {error}")
            continue
 
        # --- Problem 4: page contains no extractable text ---
        # This happens with scanned/image-only pages, blank pages, or
        # pages made entirely of vector graphics. We just skip them here;
        # OCR (needed to handle scanned pages) is a later stage.

        if not text or not text.strip():
            print(f"Warning: page {page_number} has no extractable text (skipped).")
            continue
 
        page_texts.append(text.strip())


    # Combine all page texts into a single string, one page per paragraph.
    combined_text = "\n\n".join(page_texts)

    if not combined_text:
        print(f"Warning: no extractable text found anywhere in '{file_path}'.")

    return combined_text