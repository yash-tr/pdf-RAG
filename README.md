# PDF Q&A AI System

## Overview
An AI-powered system that reads and understands PDF documents, allowing users to upload PDFs and ask questions based on their content.

## Tech Stack
- Python 3.10
- Streamlit (Web UI)
- ChromaDB (Vector Database)
- sentence-transformers (Embeddings)
- PyMuPDF (PDF Parsing)
- OpenAI or HuggingFace LLM (Answer Generation)

## Setup
1. Clone the repository
2. Create and activate the conda environment:
   ```sh
   conda create -n pdf-qa python=3.10 -y
   conda activate pdf-qa
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the app:
   ```sh
   streamlit run app.py
   ```
2. Upload PDF(s) and ask questions via the web interface.

## Features
- Upload and ingest multiple PDFs
- Chunk and embed PDF content
- Store embeddings in ChromaDB
- Natural language Q&A with RAG
- Shows source page/context for answers

## Sample PDF
See `data/sample.pdf` for a test file.

## New Features
- ChatGPT-style conversation UI using Streamlit's `st.chat_message`
- Shows source page number and highlights matched words in context
- Voice queries (Speech-to-Text powered by Google Web Speech via `SpeechRecognition`)
- Re-training / Updating PDF data with a sidebar reset button
- Docker containerization (see `Dockerfile`)
- Unit tests with **pytest** (`tests/` folder)

## Run with Docker
```bash
docker build -t pdf-qa-gemini .
docker run -p 8501:8501 --env GOOGLE_API_KEY=your_key pdf-qa-gemini
```

## Run Tests
```bash
pytest
``` 