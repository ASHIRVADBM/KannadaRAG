# 🏛️ Karnataka Heritage Chatbot (ಕರ್ನಾಟಕ ಪಾರಂಪರ್ಯ ಚಾಟ್‌ಬಾಟ್)

A **Retrieval-Augmented Generation (RAG)** based **Kannada-language chatbot** that answers cultural and historical questions about Karnataka using locally stored PDF documents.  
It integrates **OCR (for Kannada text)**, **FAISS-based semantic search**, and **Groq LLM (LLaMA 3.1)** for accurate, context-aware responses.

---

## 🚀 Features

- 🧾 Extracts text from Kannada PDFs using **PyMuPDF** and **Tesseract OCR**  
- 🔍 Stores document embeddings with **FAISS vector database**  
- 🤖 Uses **Groq LLaMA 3.1** for natural Kannada language responses  
- 🇮🇳 Kannada language detection and response enforcement  
- 💾 Persistent FAISS index for quick reloads  
- 🧠 Context-based answers via RAG pipeline  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.10+ |
| Frameworks | LangChain, SentenceTransformers |
| LLM Provider | Groq API (LLaMA 3.1) |
| Vector Store | FAISS |
| OCR | Tesseract (Kannada) |
| Optional UI | Streamlit / Flask / HTML frontend |

<img width="1904" height="883" alt="image" src="https://github.com/user-attachments/assets/0925f223-9803-4af1-8204-8a239b05f0f0" />
