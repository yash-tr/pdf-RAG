import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text

# Streamlit page configuration MUST be first
st.set_page_config(page_title="PDF Q&A AI (Gemini)", layout="wide")
st.title("📄 PDF Q&A AI System (Gemini)")

# Helper: robust rerun across Streamlit versions
def safe_rerun():
    """Call Streamlit's rerun method compatible with v1.x and v2.x."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()

# Load environment variables
load_dotenv()

# Constants
CHROMA_DB_DIR = 'chroma_db'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # You can change this

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory=CHROMA_DB_DIR
))

# Initialize embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Helper functions
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    meta = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        # Chunk by paragraphs
        for para in text.split('\n\n'):
            para = para.strip()
            if para:
                chunks.append(para)
                meta.append({'page': page_num + 1})
    return chunks, meta

def store_embeddings(chunks, meta, collection_name):
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(collection_name)
    collection = chroma_client.create_collection(collection_name, embedding_function=embedding_fn)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=meta, ids=ids)
    return collection

def query_collection(collection, query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results

def generate_answer(context_chunks, question):
    context = "\n".join(context_chunks)
    prompt = f"""
You are an AI assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:
"""
    if GEMINI_API_KEY:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    else:
        return "Gemini API key not found. Please set GOOGLE_API_KEY in your environment."

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Sidebar for retraining / updating data
with st.sidebar:
    st.header("📚 Data Management")
    if st.button("🔄 Clear & Re-Upload PDFs"):
        if 'collection' in st.session_state:
            chroma_client.delete_collection("pdf_qa")
            del st.session_state['collection']
        st.session_state['history'] = []
        safe_rerun()

# PDF Upload
st.subheader("Upload PDF(s)")
uploaded_files = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True,
                                   key="pdf_uploader")

if uploaded_files:
    all_chunks = []
    all_meta = []
    with st.spinner("Extracting and embedding PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            chunks, meta = extract_text_from_pdf(tmp_file_path)
            all_chunks.extend(chunks)
            all_meta.extend(meta)
            os.remove(tmp_file_path)
        if not all_chunks:
            st.error("❗ No text could be extracted from the uploaded PDF(s). Please ensure they contain selectable text or OCR the documents first.")
            st.stop()
        collection = store_embeddings(all_chunks, all_meta, collection_name="pdf_qa")
    st.success(f"Processed {len(uploaded_files)} PDF(s). Ready for questions!")
    st.session_state['collection'] = collection

# Function to display chat messages
def display_chat():
    for message in st.session_state['history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Only show chat UI if collection exists
if 'collection' in st.session_state:
    st.subheader("Chat with your PDFs 🗨️")
    display_chat()

    # Voice recorder (button always visible on right)
    mic_text = speech_to_text(language='en', use_container_width=False, just_once=True, key='speech_to_text')

    # Text chat input
    input_text = st.chat_input("Type your question and press Enter")

    user_question = None
    if input_text:
        user_question = input_text.strip()
    elif mic_text and mic_text != st.session_state.get('last_mic_text'):
        # Only treat new mic transcription
        st.session_state['last_mic_text'] = mic_text
        user_question = mic_text.strip()

    if user_question:
        # Append user question to history
        st.session_state['history'].append({'role': 'user', 'content': user_question})
        with st.spinner("Generating answer..."):
            results = query_collection(st.session_state['collection'], user_question, top_k=5)
            context_chunks = results['documents'][0]
            metadatas = results['metadatas'][0]
            answer = generate_answer(context_chunks, user_question)
        # Build context display with highlighting
        highlight = user_question.lower().split()
        def hl(text):
            for w in highlight:
                if len(w) > 2:
                    text = text.replace(w, f"**{w}**")
                    text = text.replace(w.capitalize(), f"**{w.capitalize()}**")
            return text
        context_display = ""
        for chunk, meta in zip(context_chunks, metadatas):
            context_display += f"\n\n**Page {meta['page']}**\n{hl(chunk)}"

        st.session_state['history'].append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
            with st.expander("Show source context"):
                st.markdown(context_display)

        # Rerun once to refresh UI with new messages
        safe_rerun()
else:
    st.info("Upload PDFs first to start chatting.") 