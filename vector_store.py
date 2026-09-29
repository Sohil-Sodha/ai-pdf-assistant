"""
Vector store module for the AI PDF Knowledge Assistant.
 
Responsible ONLY for holding text chunks together with their embedding
vectors, in memory, so they can be searched in a later stage. This
module does not create embeddings itself (see embeddings.py) and does
not implement any similarity search yet — for now, it's just storage.
"""

from dataclasses import dataclass

@dataclass
class VectorRecord:
    """One stored record: a text chunk paired with its embedding vector."""

    text: str
    embedding: list[float]

class VectorStore:
    """
    A simple in-memory store for (text, embedding) pairs.
 
    Everything lives in a plain Python list for as long as the program
    runs — there's no file persistence, no database, and no search
    capability yet. Those are separate, later stages.
    """

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, text: str, embedding: list[float]) -> None:
        """
        Store a text chunk together with its embedding vector.
 
        Args:
            text: The original text chunk (e.g. from chunker.py).
            embedding: The embedding vector already produced for that
                text (e.g. by embeddings.create_embedding()). This
                function does not generate embeddings itself.
 
        Raises:
            ValueError: If text is empty/whitespace-only, or if
                embedding is empty.
        """
        # --- Reject empty text ---
        if not text or text.strip():
            raise ValueError("Cannot store an empty text chunk.")
        
        # --- Reject an empty embedding ---
        if not embedding:
            raise ValueError("Cannot store an empty embedding.")
        
        # Text and embedding are kept together as a single record, so
        # they can never accidentally end up out of sync with each other.

        self._records.append(VectorRecord(text=text, embedding=embedding))

    def get_all(self) -> list[VectorRecord]:
        """
        Return every stored record, in the order they were added.
 
        Returns:
            A list of VectorRecord objects. Empty if nothing has been
            stored yet.
        """
        # Return a copy rather than the internal list itself, so callers
        # can't accidentally mutate the store's internal state.
        return list(self._records)
