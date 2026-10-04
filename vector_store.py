"""
Vector store module for the AI PDF Knowledge Assistant.
 
Responsible for holding text chunks together with their embedding
vectors in memory, and for comparing a query embedding against those
stored embeddings (via cosine similarity) to find the most relevant
chunks. This module does not create embeddings itself — see
embeddings.py for that — and it never calls the Gemini API.
"""

import math
from dataclasses import dataclass

@dataclass
class VectorRecord:
    """One stored record: a text chunk paired with its embedding vector."""

    text: str
    embedding: list[float]


@dataclass
class SearchResult:
    """One search result: a matched text chunk and its similarity score."""

    text: str
    score: float


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calculate the cosine similarity between two equal-length vectors.
 
    Cosine similarity measures the angle between two vectors rather
    than their length. It ranges from -1 (pointing in opposite
    directions) to 1 (pointing in exactly the same direction), with 0
    meaning the vectors are unrelated (perpendicular). For text
    embeddings, a higher score generally means more similar meaning.
    """
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    # --- Handle zero-magnitude vectors safely ---
    if magnitude_a == 0 or magnitude_b ==0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


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
    
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """
        Find the stored chunks most similar to a query embedding.
 
        Compares query_embedding against every stored embedding using
        cosine similarity, and returns the best matches ordered from
        most to least similar.
 
        Args:
            query_embedding: The embedding of the user's question,
                already created elsewhere (e.g. via
                embeddings.create_embedding()). This method does not
                generate embeddings itself.
            top_k: Maximum number of results to return.
 
        Returns:
            A list of SearchResult objects (text + similarity score),
            sorted from highest similarity to lowest, with at most
            top_k entries. Returns an empty list if the store is empty,
            if top_k <= 0, or if no stored embedding matches the query
            embedding's dimensions.
 
        Raises:
            ValueError: If query_embedding is empty.
        """
        # --- Reject an empty query embedding ---
        if not query_embedding:
            raise ValueError("Cannot search with an empty query embedding.")
        
        # --- Handle top_k <= 0 ---
        # Asking for zero (or a negative number of) results isn't an
        # error on the caller's part — there's just nothing to return.
        if top_k <= 0:
            return []
        
        results: list[SearchResult] = []

        for record in self._records:
            # --- Handle vectors with incompatible dimensions ---
            # A stored embedding from a different model (or a different
            # dimension for any reason) can't be meaningfully compared
            # to the query embedding, so it's skipped rather than
            # crashing the entire search.
            if len(record.embedding) != len(query_embedding):
                print(
                    f"Warning: skipping a stored record with "
                    f"{len(record.embedding)} dimensions; query embedding "
                    f"has {len(query_embedding)} dimensions."
                )
                continue

            score = _cosine_similarity(query_embedding, record.embedding)
            results.append(SearchResult(text=record, score=score))

        # Highest similarity first.
        results.sort(key=lambda result: result.score, reverse=True)

        return results[:top_k]
