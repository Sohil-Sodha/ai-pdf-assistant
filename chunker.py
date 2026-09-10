"""
Text chunking module for the AI PDF Knowledge Assistant.
 
Responsible ONLY for splitting a large block of text into smaller,
overlapping pieces ("chunks"). This module knows nothing about PDFs,
embeddings, or Gemini — it just takes a string in and returns a list
of smaller strings out.
 
Uses only the Python standard library.
"""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    # Split text into fixed-size, overlapping chunks.

    # --- Validate parameters first, regardless of the text content ---
    if chunk_size <=0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")
    
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}), "
            "otherwise chunks would never move forward.")
    
    # --- Edge case: nothing meaningful to chunk ---
    if not text or not text.strip():
        return []
    
    chunks: list[str] = []
    text_length = len(text)

    # How far the starting position moves forward for each new chunk.
    # This is smaller than chunk_size, which is what creates the overlap.
    step = chunk_size - overlap

    start = 0
    while start < text_length:
        end = start + chunk_size
        piece = text[start:end].strip() 

        # Skip pieces that are empty after stripping (e.g. a trailing
        # slice that was only whitespace) instead of returning junk.
        if piece:
            chunks.append(piece)

        start += step

    return chunks