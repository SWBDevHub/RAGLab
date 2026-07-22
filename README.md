# RAGLab

AI-powered document intelligence system using Retrieval-Augmented Generation (RAG). Upload a PDF. Ask questions. Get cited answers grounded in the document.

## What it does

RAGLab lets you query any PDF document using natural language and get answers that are:

- **Grounded** — Claude only answers from the document, never from general knowledge
- **Cited** — every answer shows exactly which page and chunk it came from
- **Honest** — if the answer isn't in the document, it says so rather than hallucinating

The interface includes:
- **Document library** — upload and manage multiple PDFs
- **Upload pipeline** — shows pages extracted, chunks created, and vectors stored
- **Q&A page** — ask questions, see answers with source citations and relevance scores
- **Retrieved chunks** — expandable view showing exactly what the system searched

## How RAG works

Standard AI answers from training data — it guesses. RAG answers from your document — it retrieves.

```
Upload:  PDF → extract text → split into chunks → embed as vectors → store in ChromaDB
Query:   Question → embed → find similar chunks → Claude answers from those chunks only
```

**Why this matters:** Claude cannot hallucinate content that isn't in the retrieved chunks. Every answer is traceable to a specific page in the original document.

## Why it exists

Most AI tools answer from general knowledge. RAG is the architecture used when accuracy and traceability matter — legal documents, compliance frameworks, technical specifications, internal knowledge bases. RAGLab demonstrates the full RAG pipeline from ingestion to cited retrieval.

## Demo

![RAGLab upload page](https://raw.githubusercontent.com/SWBDevHub/RAGLab/main/static/screenshots/upload.png)
![RAGLab processing page](https://raw.githubusercontent.com/SWBDevHub/RAGLab/main/static/screenshots/processed.png)
![RAGLab Q&A page](https://raw.githubusercontent.com/SWBDevHub/RAGLab/main/static/screenshots/answer.png)

## Tech stack

- **Python / Flask** — web framework
- **Anthropic Claude API** — answer generation (claude-haiku-4-5)
- **ChromaDB** — local vector database for storing and searching embeddings
- **sentence-transformers** — local embedding model (all-MiniLM-L6-v2, runs offline)
- **PyMuPDF** — PDF text extraction
- **Jinja2** — templating

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/SWBDevHub/RAGLab.git
cd RAGLab
```

**2. Install dependencies**
```bash
pip install flask anthropic python-dotenv chromadb PyMuPDF sentence-transformers
```

Note: First run downloads the embedding model (~90MB). This is a one-time download.

**3. Add your API key**

Create a `.env` file in the root:
```
ANTHROPIC_API_KEY=your_key_here
```

Get a key at [console.anthropic.com](https://console.anthropic.com)

**4. Run**
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Project structure

```
raglab/
├── app.py                          # Flask routes — upload, ask, delete
├── services/
│   ├── document_processor.py       # PDF extraction, chunking, ChromaDB storage
│   └── ai_query.py                 # Claude answer generation with citations
├── templates/
│   ├── index.html                  # Document library and upload
│   ├── upload_result.html          # Ingestion pipeline result
│   └── ask.html                    # Q&A interface with citations
├── static/
│   └── style.css                   # Dark UI styling
├── uploads/                        # Temporary PDF storage (cleared after ingestion)
└── chromadb_store/                 # Local vector database (auto-created, gitignored)
```

## Skills demonstrated

- **RAG architecture** — full pipeline from PDF ingestion to cited retrieval
- **Vector embeddings** — semantic search using sentence-transformers
- **ChromaDB** — local vector database setup and querying
- **Hallucination prevention** — grounding Claude's answers in retrieved chunks only
- **PDF processing** — text extraction and intelligent chunking with overlap
- **Citation generation** — tracing answers back to source pages
- **API integration** — Anthropic Claude API
- **Full-stack Python** — Flask, Jinja2, REST routing

## Limitations & future work

This is a v0.1 prototype built for portfolio and learning purposes.

Planned additions:
- Multiple document cross-querying
- Chunking strategy comparison (size, overlap, semantic)
- Retrieval quality evaluation and scoring
- Support for .txt, .docx formats
- Integration with ComplianceMap — ground compliance answers in official regulatory PDFs
- Embedding model comparison

## Disclaimer

RAGLab answers are grounded in uploaded documents only. Always verify important information against original sources.
