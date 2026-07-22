# =============================================
# RAGLab — AI document intelligence system
# services/document_processor.py
# Handles PDF ingestion, chunking, and vector storage
# =============================================

import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

# Initialise ChromaDB with persistent storage
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chromadb_store')
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use sentence transformers for embeddings (free, local, no API needed)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_or_create_collection(collection_name):
    """Get or create a ChromaDB collection for a document."""
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )


def list_documents():
    """List all ingested documents."""
    try:
        collections = chroma_client.list_collections()
        docs = []
        for col in collections:
            collection = chroma_client.get_collection(col.name, embedding_function=embedding_fn)
            count = collection.count()
            # Get metadata from first chunk
            results = collection.get(limit=1, include=["metadatas"])
            filename = results["metadatas"][0].get("filename", col.name) if results["metadatas"] else col.name
            docs.append({
                "collection_name": col.name,
                "filename": filename,
                "chunks": count
            })
        return docs
    except Exception as e:
        print(f"Error listing documents: {e}")
        return []


def delete_document(collection_name):
    """Delete a document collection."""
    try:
        chroma_client.delete_collection(collection_name)
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with page numbers."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": page_num,
                "text": text
            })
    doc.close()
    return pages


def chunk_text(pages, chunk_size=500, overlap=50):
    """Split pages into overlapping chunks."""
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        words = text.split()

        i = 0
        chunk_index = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text.strip()) > 50:  # skip tiny chunks
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_words)
                })
                chunk_index += 1

            i += chunk_size - overlap

    return chunks


def ingest_document(pdf_path, filename, collection_name):
    """
    Full ingestion pipeline:
    PDF → extract text → chunk → embed → store in ChromaDB
    """
    # Extract text
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return {"success": False, "error": "Could not extract text from PDF"}

    # Chunk text
    chunks = chunk_text(pages)
    if not chunks:
        return {"success": False, "error": "No text chunks generated"}

    # Store in ChromaDB
    collection = get_or_create_collection(collection_name)

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{collection_name}_chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({
            "filename": filename,
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "word_count": chunk["word_count"],
            "collection_name": collection_name
        })

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )

    return {
        "success": True,
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "collection_name": collection_name
    }


def retrieve_relevant_chunks(question, collection_name, n_results=5):
    """
    Retrieve the most relevant chunks for a question.
    """
    try:
        collection = chroma_client.get_collection(
            collection_name,
            embedding_function=embedding_fn
        )

        results = collection.query(
            query_texts=[question],
            n_results=min(n_results, collection.count())
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i] if "distances" in results else None
            relevance = round((1 - distance) * 100, 1) if distance is not None else None

            chunks.append({
                "text": doc,
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "filename": metadata.get("filename"),
                "relevance": relevance
            })

        return chunks

    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
