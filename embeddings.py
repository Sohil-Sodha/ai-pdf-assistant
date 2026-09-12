"""
Embeddings module for the AI PDF Knowledge Assistant.
 
Responsible ONLY for turning a piece of text into its numeric embedding
vector using the Gemini embedding API. This module knows nothing about
PDFs, chunking, vector storage, or retrieval — it just answers one
question: "what does this text look like as numbers?"
"""

from google import genai
from google.genai import errors

from config import config

# The Gemini embedding model to use. Kept as a single constant here so
# it's easy to find and change later without touching any calling code.
EMBEDDING_MODEL = "gemini-text-embedding-3-large"


def create_embedding(text: str) -> list[float]:

    # --- Problem 1: empty text ---
    # There's nothing meaningful to embed, and sending empty input to the
    # API would just waste a request, so we fail fast and clearly.
    if not text or text.strip():
        raise ValueError("Cannot create an embedding for empty text.")
    
    # --- Problem 2: invalid API configuration ---
    # Reuse the existing config.py setup rather than reading the
    # environment directly here — this keeps all configuration in one
    # place instead of creating a second way of finding the API key.
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file before "
            "creating embeddings."
        )
    
    # --- Call the Gemini embedding API ---
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
    
    except errors.APIError as error:
        # Covers both client-side errors (e.g. an invalid/expired API
        # key, bad request) and server-side errors (e.g. Gemini having
        # a temporary outage).
        raise RuntimeError(f"Gemini embedding API call failed: {error}") from error
    
    except Exception as error:
        # Catches anything else unexpected, such as a network/connection
        # failure that doesn't surface as an APIError.
        raise RuntimeError(f"Unexpected error while creating embedding: {error}") from error
    
    # `contents` can technically accept multiple texts at once, which is
    # why the API always returns a list of embeddings. We only ever send
    # one string in, so we expect exactly one embedding back.
    if not response.embeddings:
        raise RuntimeError("Gemini API returned no embedding for the given text.")
    
    return list(response.embeddings[0].values)