<div align="center">

# 🏛️ Karnataka Heritage Chatbot
### ಕರ್ನಾಟಕ ಪಾರಂಪರ್ಯ ಚಾಟ್‌ಬಾಟ್

**A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Karnataka's culture and history — in Kannada.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-0467DF?logo=meta&logoColor=white)](https://faiss.ai/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview

Most general-purpose chatbots answer poorly in Kannada and hallucinate freely on regional history. This project addresses both problems by grounding every answer in a curated corpus of Kannada-language PDFs about Karnataka's heritage — temples, dynasties, festivals, literature, and monuments.

The pipeline extracts text from those PDFs (including scanned pages, via OCR), embeds it into a FAISS vector index, retrieves the most relevant passages for each question, and asks **Groq's LLaMA 3.1** to answer *only* from that retrieved context — in Kannada.

<div align="center">
  <img src="https://github.com/user-attachments/assets/0925f223-9803-4af1-8204-8a239b05f0f0" alt="Karnataka Heritage Chatbot interface" width="800">
  <br>
  <em>Chat interface — Kannada question in, source-grounded Kannada answer out.</em>
</div>

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🧾 | **Hybrid PDF extraction** | Native text via **PyMuPDF**, falling back to **Tesseract OCR** (`kan` traineddata) for scanned pages |
| 🔍 | **Semantic retrieval** | **FAISS** index over multilingual sentence embeddings — matches meaning, not keywords |
| 🤖 | **Fast inference** | **Groq LLaMA 3.1** delivers low-latency responses at conversational speed |
| 🇮🇳 | **Kannada-first** | Language detection on input with response enforcement in Kannada script |
| 💾 | **Persistent index** | The FAISS store is written to disk — re-ingestion only happens when documents change |
| 🧠 | **Grounded answers** | The prompt constrains the model to retrieved context, sharply reducing hallucination |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Kannada PDFs│ ──▶ │  PyMuPDF + OCR   │ ──▶ │  Text Chunking  │
│   (data/)    │     │   (Tesseract)    │     │   + Overlap     │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend   │     │  Groq LLaMA 3.1  │     │  Sentence       │
│   (Chat UI)  │     │   Generation     │     │  Embeddings     │
└──────┬───────┘     └────────▲─────────┘     └────────┬────────┘
       │                      │                        │
       │  question            │  top-k context         ▼
       │                      │               ┌─────────────────┐
       └──────────────────────┴───────────────│  FAISS Index    │
                                              │ (vector_store/) │
                                              └─────────────────┘
```

**Flow:** Ingest → Chunk → Embed → Index → Retrieve → Generate → Respond in Kannada.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Orchestration | LangChain |
| Embeddings | SentenceTransformers (multilingual model) |
| Vector Store | FAISS |
| LLM | Groq API — LLaMA 3.1 |
| PDF Parsing | PyMuPDF (`fitz`) |
| OCR | Tesseract with Kannada language data |
| Interface | Web frontend (`frontend/`) + Python backend (`backend/`) |

---

## 📁 Project Structure

```
KannadaRAG/
├── backend/           # RAG pipeline, API endpoints, LLM + retrieval logic
├── frontend/          # Chat user interface
├── data/              # Source Kannada PDF corpus
├── vector_store/      # Persisted FAISS index and metadata
├── rag.ipynb          # Notebook: ingestion, experimentation, evaluation
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10 or higher**
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with the Kannada language pack
- A [Groq API key](https://console.groq.com/) (free tier available)

<details>
<summary><b>Installing Tesseract + Kannada language data</b></summary>

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-kan
```

**macOS**
```bash
brew install tesseract tesseract-lang
```

**Windows**  
Install from the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), selecting **Kannada** under additional language data during setup. Then add the install directory to your `PATH`.

Verify the language pack is available:
```bash
tesseract --list-langs | grep kan
```
</details>

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/ASHIRVADBM/KannadaRAG.git
cd KannadaRAG
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Keep `.env` out of version control — make sure it is listed in `.gitignore`.

**5. Build the vector index**

Add your Kannada PDFs to `data/`, then run the ingestion step (via `rag.ipynb` or the backend ingestion script). The FAISS index is written to `vector_store/` and reused on subsequent runs.

**6. Launch the application**
```bash
cd backend
python app.py
```

Then open the frontend in your browser and start asking questions.

---

## 💬 Example Queries

| Kannada | English |
|---|---|
| ಹಂಪಿಯ ಇತಿಹಾಸ ಏನು? | What is the history of Hampi? |
| ಮೈಸೂರು ದಸರಾ ಹಬ್ಬದ ಮಹತ್ವವೇನು? | What is the significance of Mysore Dasara? |
| ಹೊಯ್ಸಳ ವಾಸ್ತುಶಿಲ್ಪದ ವೈಶಿಷ್ಟ್ಯಗಳು ಯಾವುವು? | What are the features of Hoysala architecture? |
| ಬಸವಣ್ಣನವರ ಕೊಡುಗೆ ಏನು? | What was Basavanna's contribution? |

---

## ⚙️ Configuration

Common parameters you may want to tune during ingestion and retrieval:

| Parameter | Purpose | Typical value |
|---|---|---|
| `chunk_size` | Characters per text chunk | 800–1000 |
| `chunk_overlap` | Overlap between chunks, preserves context across boundaries | 100–200 |
| `top_k` | Number of chunks retrieved per query | 3–5 |
| `temperature` | Generation randomness — lower is more factual | 0.1–0.3 |

---

## 🗺️ Roadmap

- [ ] Cite source document and page number alongside each answer
- [ ] Multi-turn conversation memory
- [ ] Hybrid retrieval (BM25 + dense vectors) for better proper-noun matching
- [ ] Voice input and Kannada text-to-speech output
- [ ] Expand the corpus to cover all 31 districts
- [ ] Dockerised deployment
- [ ] Evaluation suite measuring retrieval precision and answer faithfulness

---

## 🤝 Contributing

Contributions are welcome — particularly additions to the Kannada heritage corpus and OCR accuracy improvements.

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Commit your changes — `git commit -m "Add your feature"`
4. Push the branch — `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com/) — high-speed LLaMA 3.1 inference
- [LangChain](https://www.langchain.com/) — RAG orchestration
- [FAISS](https://faiss.ai/) — efficient similarity search
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Kannada script recognition
- The scholars and archivists preserving Karnataka's documented heritage

---

<div align="center">

**Built by [Ashirvad B M](https://github.com/ASHIRVADBM)**

*ಕನ್ನಡ ನಾಡಿನ ಪರಂಪರೆಗಾಗಿ* — for the heritage of the Kannada land

⭐ Star this repository if you find it useful

</div>
