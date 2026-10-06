"""
AI PDF Knowledge Assistant
Stage 9: Full RAG pipeline (CLI orchestration)
 
This is the entry point and orchestration layer of the application. It
wires together the already-built pipeline modules - PDF loading,
chunking, embedding, vector storage, retrieval, and Gemini answer
generation - into one interactive command-line session.
 
app.py does not implement any of that logic itself. It only calls each
module's existing public functions/classes, in order, and handles the
user interaction (prompts, input, printed output) around them.
"""

from pdf_loader import load_pdf
from chunker import chunk_text
from embeddings import create_embedding
from vector_store import VectorStore
from retriever import Retriever
from gemini import generate_answer

def main():
    """Application entry point: load a PDF, then answer questions about it."""

    print()
    print("=" * 55)
    print("  Welcome to the AI PDF Knowledge Assistant")
    print("=" * 55)
    print()
    print("Ask questions about a PDF document - answered by Gemini,")
    print("grounded in that document's own content.")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load the PDF and split it into chunks.
    # Loops on recoverable problems (bad path, no text, no chunks)
    # instead of crashing, so a typo doesn't end the whole session.
    # ------------------------------------------------------------------
    chunks: list[str] = []

    while True:
        file_path = input("Enter the path to a PDF file: ").strip()

        try:
            text =load_pdf(file_path)

        except (FileNotFoundError, ValueError) as error:
            print(f"Could not load that PDF: {error}")
            print("Please try again.\n")
            continue

        if not text.strip():
            print("That PDF has no extractable text (it may be scanned/image-only).")
            print("Please try a different PDF.\n")
            continue

        chunks = chunk_text(text)

        if not chunks:
            print("No usable text chunks could be created from that PDF.")
            print("Please try a different PDF.\n")
            continue

        break # Success: we have usable chunks, move on.

    print(f"\nLoaded the PDF and split it into {len(chunks)} chunk(s).")


    # ------------------------------------------------------------------
    # Step 2: Embed each chunk and store it in the VectorStore.
    # ------------------------------------------------------------------
    vector_store = VectorStore()
    embedded_count = 0

    print("Generating embeddings... (this may take a moment)")
    for chunk in chunks:
        if not chunk.strip():
            # chunk_text() already filters these out, but checking again
            # here costs nothing and keeps this loop safe on its own.
            continue

        try:
            embedding = create_embedding(chunk)

        except ValueError as error:
            # Every chunk reaching this point is already non-empty, so a
            # ValueError here can only mean the Gemini API key itself is
            # missing or invalid. That won't fix itself on the next
            # chunk, so stop now instead of repeating the same failure.
            print(f"\nError: {error}")
            print("Cannot continue without a valid Gemini API key.")
            return
        
        except RuntimeError as error:
            # A single chunk failed to embed (e.g. a transient network
            # or API issue). Skip just this one chunk and keep going -
            # losing one chunk out of many still leaves a usable store.
            print(f"Warning: skipping a chunk due to an embedding error: {error}")
            continue

        vector_store.add(chunk, embedding)
        embedded_count += 1

    if embedded_count == 0:
        print("\nError: no chunk could be embedded. Cannot answer questions about this PDF.")
        return
    
    print(f"Stored {embedded_count} chunk(s) in the vector store.\n")


    # ------------------------------------------------------------------
    # Step 3: Question-and-answer loop.
    # ------------------------------------------------------------------
    retriever = Retriever(vector_store)

    print("You can now ask questions about the PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question (or type 'exit' to quit).\n")
            continue

        # --- Retrieve the most relevant chunks for this question ---
        try:
            results = retriever.retrieve(question, top_k=3)

        except ValueError as error:
            print(f"Error: {error}\n")
            continue

        except RuntimeError as error:
            print(f"Error: could not retrieve relevant content: {error}\n")
            continue

        # VectorStore.search() (used internally by Retriever) currently
        # returns a list of SearchResult objects, each with a .text and
        # a .score. We only need the text for generate_answer().
        context_chunks = [result.text for result in results]

        # --- Generate an answer grounded in the retrieved chunks ---
        try:
            answer = generate_answer(question, context_chunks)

        except ValueError as error:
            print(f"Error: {error}\n")
            continue

        except RuntimeError as error:
            print(f"Error: could not generate an answer: {error}\n")
            continue

        print(f"\nAnswer: {answer}\n")



if __name__ == "__main__":
    main()