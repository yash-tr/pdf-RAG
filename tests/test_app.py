import tempfile, os
from app import extract_text_from_pdf, store_embeddings, chroma_client

def test_extract_text_from_sample_pdf():
    # Create a small sample PDF dynamically
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50,50), "Hello World")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        doc.save(tmp.name)
        tmp_path = tmp.name
    chunks, meta = extract_text_from_pdf(tmp_path)
    os.remove(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].startswith("Hello")
    assert meta[0]['page'] == 1

def test_store_embeddings_length():
    chunks = ["sample text 1", "sample text 2"]
    meta = [{'page':1}, {'page':1}]
    collection = store_embeddings(chunks, meta, collection_name="test_collection")
    assert collection.count() == 2
    # cleanup
    chroma_client.delete_collection("test_collection") 