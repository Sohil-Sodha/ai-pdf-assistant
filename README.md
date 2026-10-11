# AI PDF Knowledge Assistant

A command-line application that lets you ask questions about a PDF and get
answers based on the PDF's own content, using **Retrieval-Augmented
Generation (RAG)** with the Google Gemini API.

## How it works

```
Start up (once per PDF)                    For every question
-----------------------                    ------------------
PDF file                                   Your question
  -> extract text (pypdf)                    -> embed the question (Gemini)
  -> split into overlapping chunks           -> find the 3 most similar chunks
  -> embed each chunk (Gemini)                  (cosine similarity, in memory)
  -> keep chunks + vectors in memory         -> send question + those chunks to Gemini
                                             -> print the answer
```

The PDF is embedded **once** when you start the app. Each question after
that costs one embedding request (for the question) plus one answer request.

## Requirements

- **Python 3.10 or newer**
- A **Gemini API key** (create one in [Google AI Studio](https://aistudio.google.com/))
- An internet connection (the app calls the Gemini API)

## Setup

1. Open a terminal in the project folder (the one containing `app.py`).

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a file named `.env` in the project folder (next to `app.py`) with
   your API key on a single line, with no quotes and no spaces around `=`:

   ```
   GEMINI_API_KEY=your-key-here
   ```

   > `.env` is listed in `.gitignore` so it is never committed. That also
   > means a fresh `git clone` does **not** contain it - you have to create it.
   > Alternatively, set `GEMINI_API_KEY` as a normal environment variable
   > (a real environment variable takes priority over `.env`).

5. Run the app:

   ```bash
   python app.py
   ```

## Using it

```
Enter the path to a PDF file (or 'exit' to quit): ~/Documents/report.pdf

Loaded the PDF and split it into 12 chunk(s).
Generating embeddings... (this may take a moment)
Stored 12 of 12 chunk(s) in the vector store.

You can now ask questions about the PDF.
Type 'exit' or 'quit' to stop.

Your question: What were the main findings?

Answer: ...
```

- Paths can be pasted with quotes and may start with `~`.
- Type `exit` or `quit` (or press Ctrl+C) to leave at any prompt.
- If the answer is not in the retrieved text, the assistant is instructed to
  say so rather than guess.

## Project structure

| File | Responsibility |
|---|---|
| `app.py` | Command-line interface; runs the steps above in order |
| `config.py` | Reads `GEMINI_API_KEY` from `.env` / the environment |
| `pdf_loader.py` | `load_pdf()` - extracts text from a PDF |
| `chunker.py` | `chunk_text()` - splits text into overlapping chunks |
| `embeddings.py` | `create_embedding()` - turns text into a vector via Gemini |
| `vector_store.py` | `VectorStore` - in-memory storage and cosine-similarity search |
| `retriever.py` | `Retriever` - embeds a question and searches the store |
| `gemini.py` | `generate_answer()` - asks Gemini to answer from the retrieved chunks |

## Configuration

| What | Where | Default |
|---|---|---|
| API key | `.env` or environment variable `GEMINI_API_KEY` | none (required) |
| Embedding model | `EMBEDDING_MODEL` in `embeddings.py` | `gemini-embedding-001` |
| Answer model | `GENERATION_MODEL` in `gemini.py` | `gemini-2.5-flash` |
| Chunk size / overlap | `chunk_text()` defaults in `chunker.py` | 1000 / 200 characters |
| Chunks sent to Gemini per question | `top_k=3` in `app.py` | 3 |

Model names are set by Google and can change over time. If you see a
"model not found" style error, update the constant to a model your key can use.

## Current limitations

- **Nothing is saved.** Vectors live in memory, so the PDF is re-processed (and
  re-embedded, using API requests) every time you start the app. One PDF per session.
- **Text-based PDFs only.** There is no OCR, so scanned/image-only PDFs report
  "no extractable text". Tables and multi-column layouts may extract awkwardly.
- **Password-protected PDFs are not supported.** (PDFs that only restrict
  copying/printing, and open without a password, do work.)
- **Simple chunking.** Text is split by character count, so a chunk can start or
  end in the middle of a word or sentence.
- **Only 3 chunks are used per question.** Answers that depend on information
  spread across many parts of the document may be incomplete, and answers do not
  cite page numbers.
- **One API request per chunk.** Large PDFs take a while and may hit your API
  quota or rate limit. If 3 embedding requests fail in a row, the app stops and
  tells you to check your key, connection, and quota.
- **Your data is sent to Google.** PDF text and your questions are sent to the
  Gemini API. Don't use this with documents you are not allowed to share.
- **Answers can still be wrong.** Check important answers against the PDF.
- **No automated test suite is included.**

## Troubleshooting

| Message | What to do |
|---|---|
| `GEMINI_API_KEY is not set` | Create `.env` as described in Setup (the file must be named exactly `.env`). |
| `400 ... API key not valid` | The key is wrong or was revoked; create a new one. |
| `429 ... quota` / rate-limit errors | Wait and retry, or check your plan's quota. |
| `404 ... model ... not found` | Update `EMBEDDING_MODEL` / `GENERATION_MODEL` (see Configuration). |
| `That PDF has no extractable text` | The PDF is probably a scan; OCR is not supported. |
| `Gemini returned an empty response` | The reply was blocked or empty; try rephrasing the question. |

## Security

Never commit your API key. `.env` is git-ignored; if a key is ever committed or
shared by accident, revoke it in Google AI Studio and create a new one.