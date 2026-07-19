import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import re
import chromadb

# shared embedding model — loaded ONCE here, imported by retriever
model = SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def clean_text(text):
    pos = text.find("Abstract")
    if pos != -1:
        text = text[pos + len("Abstract"):]
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub("[∗†‡]", "", text)
    return text


def chunk_text(text, size, overlap):
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
    return chunks


# --- run the ingestion pipeline (module-level: runs once on import) ---
paper = read_pdf(os.path.join(ROOT, "data", "papers", "attention.pdf"))
cleaned = clean_text(paper)
chunks = chunk_text(cleaned, size=120, overlap=20)

# --- Chroma vector store ---
client = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))
collection = client.get_or_create_collection(
    name="attention_paper",
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [model.encode(c).tolist() for c in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks)
    print(f"added {len(chunks)} chunks to Chroma")
else:
    print(f"collection already has {collection.count()} chunks")