# =============================================
# RAGLab — AI document intelligence system
# services/ai_query.py
# Handles answer generation with citations
# =============================================

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))


def generate_answer(question, chunks, filename):
    """
    Generate a grounded answer from retrieved chunks.
    Returns answer with citations.
    """

    if not chunks:
        return {
            "answer": "I could not find any relevant information in the document to answer this question.",
            "citations": [],
            "confidence": "none",
            "found_in_document": False
        }

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} — Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise document analysis assistant. Answer the question below using ONLY the document excerpts provided.

Document: {filename}
Question: {question}

Document excerpts:
{context}

Rules:
1. Answer ONLY from the provided excerpts — do not use outside knowledge
2. If the answer is not in the excerpts, say exactly: "This information was not found in the document."
3. Always cite your sources using [Source N] notation
4. Be specific and quote directly where helpful
5. If multiple sources support the answer, cite all of them

Answer:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    answer_text = response.content[0].text

    # Determine if answer was found
    not_found_phrases = [
        "not found in the document",
        "not mentioned in",
        "no information",
        "cannot find",
        "not provided in"
    ]
    found = not any(phrase in answer_text.lower() for phrase in not_found_phrases)

    # Build citations
    citations = []
    for i, chunk in enumerate(chunks, 1):
        if f"[Source {i}]" in answer_text:
            citations.append({
                "source_num": i,
                "page": chunk["page"],
                "text": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                "relevance": chunk.get("relevance")
            })

    # Confidence based on relevance scores
    if chunks and chunks[0].get("relevance"):
        top_relevance = chunks[0]["relevance"]
        if top_relevance > 70:
            confidence = "high"
        elif top_relevance > 50:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "medium"

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence": confidence,
        "found_in_document": found,
        "chunks_searched": len(chunks)
    }
