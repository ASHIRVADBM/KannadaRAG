from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, uvicorn
from rag_core import build_index, rag_simple, groq_invoke

app = FastAPI(title="Kannada RAG Chatbot")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Build FAISS index once at startup
print("🔧 Building vector index from PDFs...")
retriever, chunks = build_index(DATA_DIR)
print("✅ Index ready!")

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    if not chunks:
        return JSONResponse({"error": "No data found. Please add PDFs in backend/data"})
    answer = rag_simple(query, retriever, chunks, groq_invoke)
    return answer

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
