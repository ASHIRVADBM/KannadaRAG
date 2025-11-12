import os, json, re, fitz, pytesseract, faiss, hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# === Load environment ===
load_dotenv()

# === Config ===
DATA_DIR = "./data"
FAISS_DIR = "./vector_store"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

TESSERACT_LANG = "kan"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MIN_SCORE = 0.55
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Data Structure ===
@dataclass
class Doc:
    page_content: str
    metadata: dict

# === PDF Extraction ===
def extract_text_from_pdf(pdf_path):
    docs = []
    pdf = fitz.open(pdf_path)
    for i, page in enumerate(pdf):
        text = page.get_text("text").strip()
        used_ocr = False
        if len(text) < 50:
            from PIL import Image
            import io
            pix = page.get_pixmap(dpi=200)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(image, lang=TESSERACT_LANG).strip()
            if len(ocr_text) > len(text):
                text, used_ocr = ocr_text, True
        docs.append(Doc(text, {"source": os.path.basename(pdf_path), "page": i + 1, "ocr": used_ocr}))
    pdf.close()
    return docs


def process_all_pdfs(folder):
    pdfs = list(Path(folder).glob("*.pdf"))
    all_docs = []
    for p in pdfs:
        print(f"📘 Processing {p.name}")
        all_docs.extend(extract_text_from_pdf(str(p)))
    print(f"✅ Loaded {len(all_docs)} pages total.")
    return all_docs


# === Text Splitting ===
def split_docs(docs, size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    lang_docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]
    chunks = splitter.split_documents(lang_docs)
    print(f"🧩 Split into {len(chunks)} chunks.")
    return [Doc(c.page_content, c.metadata) for c in chunks]


# === Embeddings + FAISS ===
class EmbeddingManager:
    def __init__(self):
        print(f"🔡 Loading embedding model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)


class FaissVectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []

    def add(self, docs, embs):
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        self.index.add(embs)
        self.meta.extend([d.metadata for d in docs])
        faiss.write_index(self.index, os.path.join(FAISS_DIR, "faiss.index"))
        with open(os.path.join(FAISS_DIR, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        print(f"✅ Added {len(docs)} vectors to FAISS index.")

    def load(self):
        try:
            index_path = os.path.join(FAISS_DIR, "faiss.index")
            meta_path = os.path.join(FAISS_DIR, "meta.json")
            if os.path.exists(index_path) and os.path.exists(meta_path):
                self.index = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
                print(f"📂 Loaded existing FAISS index ({len(self.meta)} vectors).")
                return True
        except Exception as e:
            print("⚠️ Failed to load FAISS index:", e)
        return False

    def search(self, q_emb, top_k=3):
        q_emb = q_emb / np.linalg.norm(q_emb)
        D, I = self.index.search(np.array([q_emb]), top_k)
        return [{"score": float(D[0][i]), "meta": self.meta[I[0][i]]} for i in range(top_k) if I[0][i] < len(self.meta)]


class Retriever:
    def __init__(self, emb_mgr, vs):
        self.emb_mgr, self.vs = emb_mgr, vs

    def retrieve(self, query, top_k=3):
        emb = self.emb_mgr.embed([query])[0]
        results = self.vs.search(emb, top_k)
        return [r for r in results if r["score"] >= MIN_SCORE]


# === LLM ===
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=512,
)


def groq_invoke(prompt):
    try:
        resp = llm.invoke([prompt])
        return resp.content.strip()
    except Exception as e:
        return f"⚠️ LLM Error: {e}"


# === Kannada Detection ===
def is_kannada_text(text):
    return len(re.findall(r"[\u0C80-\u0CFF]", text)) > 0


# === Build / Load Index ===
def build_index(folder):
    emb_mgr = EmbeddingManager()
    vs = FaissVectorStore(emb_mgr.dim)

    if vs.load():
        retriever = Retriever(emb_mgr, vs)
        chunks = [Doc("", {})]
        return retriever, chunks

    docs = process_all_pdfs(folder)
    chunks = split_docs(docs)
    embs = emb_mgr.embed([c.page_content for c in chunks])
    vs.add(chunks, embs)
    retriever = Retriever(emb_mgr, vs)
    return retriever, chunks


# === RAG ===
def rag_simple(query, retriever, chunks, llm_invoke, top_k=3):
    if not is_kannada_text(query):
        return {"status": "invalid_language", "answer": "ದಯವಿಟ್ಟು ಕನ್ನಡದಲ್ಲಿ ಪ್ರಶ್ನೆ ಕೇಳಿ.", "sources": []}

    results = retriever.retrieve(query, top_k)
    if not results:
        return {"status": "no_context", "answer": "⚠️ ಯಾವುದೇ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ.", "sources": []}

    context = "\n\n".join([r["meta"].get("source", "") for r in results])
    prompt = f"""
    ಕೆಳಗಿನ ಪಾರ್ಶ್ವಭೂಮಿಯ ಆಧಾರದ ಮೇಲೆ ಮಾತ್ರ ಪ್ರಶ್ನೆಗೆ ನಿಖರ ಮತ್ತು ಸಂಕ್ಷಿಪ್ತ ಉತ್ತರ ನೀಡಿ.
    ಪಾರ್ಶ್ವಭೂಮಿಯ ಹೊರಗಿನ ಯಾವುದೇ ಮಾಹಿತಿ ಸೇರಿಸಬೇಡಿ.
    ಉತ್ತರವು 2 ಅಥವಾ 3 ವಾಕ್ಯಗಳಲ್ಲಿ ಇರಲಿ.

    ಪ್ರಶ್ನೆ: {query}

    ಪಾರ್ಶ್ವಭೂಮಿ:
    {context}

    ಉತ್ತರ:
    """
    response = llm_invoke(prompt)
    return {"status": "ok", "answer": response, "sources": results}
