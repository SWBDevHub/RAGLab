# =============================================
# RAGLab — AI document intelligence system
# app.py
# =============================================

import os
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from services.document_processor import ingest_document, list_documents, delete_document, retrieve_relevant_chunks
from services.ai_query import generate_answer

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def make_collection_name(filename):
    """Convert filename to valid ChromaDB collection name."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name[:50]
    if len(name) < 3:
        name = name + "_doc"
    return name.lower()


@app.route("/", methods=["GET"])
def index():
    documents = list_documents()
    return render_template("index.html", documents=documents)


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    collection_name = make_collection_name(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = ingest_document(filepath, filename, collection_name)

    # Clean up uploaded file after ingestion
    if os.path.exists(filepath):
        os.remove(filepath)

    return render_template("upload_result.html", result=result)


@app.route("/ask/<collection_name>", methods=["GET", "POST"])
def ask(collection_name):
    documents = list_documents()
    current_doc = next((d for d in documents if d["collection_name"] == collection_name), None)

    if not current_doc:
        return redirect(url_for('index'))

    result = None
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            chunks = retrieve_relevant_chunks(question, collection_name)
            result = generate_answer(question, chunks, current_doc["filename"])
            result["retrieved_chunks"] = chunks

    return render_template(
        "ask.html",
        current_doc=current_doc,
        documents=documents,
        result=result,
        question=question
    )


@app.route("/delete/<collection_name>", methods=["POST"])
def delete(collection_name):
    delete_document(collection_name)
    return redirect(url_for('index'))


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
