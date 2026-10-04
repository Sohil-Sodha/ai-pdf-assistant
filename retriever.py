"""
Retriever module for the AI PDF Knowledge Assistant.
 
Responsible ONLY for coordinating the retrieval process: turning a
user's question into an embedding (via embeddings.py) and asking an
existing VectorStore to find the most similar stored chunks (via
vector_store.py). This module does not create embeddings itself, does
not implement similarity search itself, and does not generate answers
— it just wires those two existing pieces together.
"""

from embeddings import create_embedding
from vector_store import SearchResult, VectorStore

class Retriever:
    """
    Coordinates retrieval: question -> embedding -> vector store search.
 
    Holds a reference to an already-built VectorStore (populated
    elsewhere, e.g. after PDF loading, chunking, and embedding) and
    uses the existing create_embedding() function to turn a user's
    question into a vector it can search with.
    """
    def __init__(self, vector_store: VectorStore):
        """
        Args:
            vector_store: An existing, already-populated VectorStore
                instance to search against.
        """
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Find the stored chunks most relevant to a user's question.
 
        Args:
            query: The user's question, as plain text.
            top_k: Maximum number of relevant chunks to return.
 
        Returns:
            A list of SearchResult objects (text + similarity score),
            most relevant first. Returns an empty list if top_k <= 0.
 
        Raises:
            ValueError: If query is empty/whitespace-only, or if
                creating its embedding fails due to invalid
                configuration (e.g. a missing Gemini API key).
            RuntimeError: If creating the embedding fails due to an
                API/connection problem, or if the vector store search
                itself fails unexpectedly.
        """
        # --- Validate the query ---
        if not query or not query.strip():
            raise ValueError("Cannot retrieve results for an empty query.")
        
        # --- Handle an invalid top_k early ---
        # There's no point spending an embedding API call if the caller
        # has asked for zero (or fewer) results anyway.
        if top_k <= 0:
            return []
        
        # --- Create an embedding for the query ---
        # This reuses embeddings.py's create_embedding() rather than
        # duplicating any embedding-generation logic here. Any error it
        # raises (ValueError for bad config, RuntimeError for an
        # API/connection failure) already has a clear message, so it's
        # allowed to propagate as-is.
        query_embedding = create_embedding(query)

        # --- Search the vector store ---
        # This reuses VectorStore.search() rather than duplicating any
        # cosine-similarity or ranking logic here.
        try:
            return self._vector_store.search(query_embedding, top_k=top_k)
        
        except Exception as error:
            # Wrap anything unexpected from the search step so the
            # caller gets a clear "retrieval failed" message instead of
            # an unrelated-looking raw traceback.
            raise RuntimeError(f"Retrieval failed while searching the vector store: {error}") from error
