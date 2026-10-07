"""
Gemini answer-generation module for the AI PDF Knowledge Assistant.
 
Responsible ONLY for sending a user's question, together with already-
retrieved PDF context, to Gemini and returning the generated answer as
plain text. This module does not load PDFs, chunk text, create
embeddings, search a vector store, or orchestrate retrieval — it just
receives already-retrieved text chunks and answers one question:
"what does Gemini say, given this question and this context?"
"""

from google import genai
from google.genai import errors

from config import config

# The Gemini model used for answer generation. Kept as a single
# constant, separate from embeddings.py's EMBEDDING_MODEL, since
# generation and embedding are different kinds of models.
GENERATION_MODEL = "gemini-3.5-flash-lite"

def generate_answer(question: str, context: list[str]) -> str:
    """
    Generate an answer to a question using Gemini, grounded in context.
    """
    # --- Reject an empty question ---
    if not question or not question.strip():
        raise ValueError("Cannot generate an answer for an empty question.")
    
    # --- Ignore empty/whitespace-only context chunks ---
    usable_chunks = [chunk.strip() for chunk in context if chunk and chunk.strip()]

    # --- Handle an empty context list appropriately ---
    # With nothing usable to ground an answer in, calling Gemini would
    # only risk an invented answer — so we skip the API call (and the
    # API key check below) entirely, and give the same honest response
    # the prompt would otherwise ask Gemini to give.
    if not usable_chunks:
        return "The answer could not be found in the provided document context."
    
    # --- Invalid API configuration ---
    # Only checked once we know we actually need to call Gemini. Reuses
    # the existing config.py setup, exactly like embeddings.py does,
    # rather than creating a second way of finding the API key.
    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to your .env file before "
            "generating answers."
        )
    
    prompt = _build_prompt(question, usable_chunks)

    # --- Call the Gemini API ---
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
    
    except errors.APIError as error:
        # Covers both client-side errors (e.g. an invalid/expired API
        # key, bad request) and server-side errors (e.g. Gemini having
        # a temporary outage).
        raise RuntimeError(f"Gemini answer generation failed: {error}") from error
    
    except Exception as error:
        # Catches anything else unexpected, such as a network/connection
        # failure that doesn't surface as an APIError.
        raise RuntimeError(f"Unexpected error while generating an answer: {error}") from error
    
    return response.text


def _build_prompt(question: str, chunks: list[str]) -> str:
    """
    Build the full prompt sent to Gemini: instructions, the retrieved
    context (clearly labeled), and the user's question.
    """
    # Number each excerpt so the context section is clearly structured,
    # rather than one unbroken wall of text.
    context_section = "\n\n".join(f"[Excerpt {i}]\n{chunk}" for i, chunk in enumerate(chunks, start=1))

    return (
        "You are answering a question using ONLY the document excerpts "
        "provided below.\n\n"
        "Rules:\n"
        "- Use the excerpts as your primary source of truth.\n"
        "- Do not invent or assume information that is not supported by "
        "the excerpts.\n"
        "- If the excerpts do not contain enough information to answer "
        "the question, clearly say that the answer could not be found "
        "in the provided document context.\n"
        "- Keep your answer clear and concise.\n\n"
        f"Document excerpts:\n{context_section}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
