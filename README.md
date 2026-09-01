# AI PDF Knowledge Assistant

A Python CLI application that will eventually let you ask questions about
PDF documents using Retrieval-Augmented Generation (RAG), powered by the
Gemini API.

## Current Status: Stage 1 — Project Foundation

This stage only sets up the basic project skeleton. There is **no**:
- PDF processing
- Embeddings
- Vector database
- Retrieval logic
- RAG pipeline

Those will be added in later stages, once the foundation is solid.

## Project Structure

```
ai-pdf-assistant/
│
├── app.py             # Entry point (main())
├── config.py          # Loads config/env vars for later Gemini API use
├── .env                # Local secrets (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Gemini API key to `.env`:
   ```
   GEMINI_API_KEY=your-key-here
   ```

4. Run the app:
   ```bash
   python app.py
   ```