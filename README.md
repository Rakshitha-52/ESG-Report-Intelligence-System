# 🌱 ESG Report Intelligence System

AI-powered **Retrieval-Augmented Generation (RAG)** system for analyzing Environmental, Social, and Governance (ESG) reports with **grounded, cited answers** and **cross-company comparison**.

**Live Demo:** [(https://esg-report-intelligence-system-l3kscjdsrpjtsiewprptwu.streamlit.app/)]

---
![ESG Report Intelligence System](App_preview/Preview1.png)

# Overview

Upload ESG/Sustainability PDF reports and ask natural language questions such as:

- What is the carbon neutrality target?
- How much renewable energy is being used?
- Compare Apple's and Google's approach to emissions reduction.

Every answer is grounded in the uploaded reports and includes citations (filename + page number). If the required information is not available, the system responds with **"I don't have enough information"** instead of generating unsupported answers.

---


# Key Features

- 📄 Multi-PDF upload and processing
- 🔍 Semantic search using **FAISS** and **Sentence-BERT embeddings**
- 🤖 Natural language question answering powered by **Gemini 2.5 Flash**
- 📚 Source citations displayed as filename and page number
- ⚖️ Automatic cross-company comparison with entity-aware retrieval
- ⚙️ Adjustable retrieval depth (**k**) for better search quality

---

# Architecture

```text
                 PDF Upload
                      │
                      ▼
        Text Extraction (pdfplumber)
                      │
                      ▼
           Cleaning & Preprocessing
                      │
                      ▼
      Chunking (1000 characters, 200 overlap)
                      │
                      ▼
    Sentence-BERT Embeddings (all-MiniLM-L6-v2)
                      │
                      ▼
             FAISS Vector Store
                      │
                      ▼
                  User Query
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
Standard Retrieval      Comparison Detection
                                 │
                                 ▼
                     Per-Entity Retrieval
                      │
        └─────────────┬─────────────┘
                      ▼
      Gemini 2.5 Flash (Temperature = 0.1)
                      │
                      ▼
      Grounded Answer + Source Citations
                      │
                      ▼
                Streamlit Interface
```

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| PDF Processing | pdfplumber, PyPDF2 (fallback) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Database | FAISS |
| LLM | Google Gemini 2.5 Flash |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

# Installation

```bash
git clone https://github.com/Rakshitha-52/ESG-Report-Intelligence-System.git

cd ESG-Report-Intelligence-System

python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Add your GOOGLE_API_KEY

streamlit run app.py
```

> **Note:** Sample ESG reports used during development are **not included** in this repository due to their size. They can be downloaded from the official sustainability pages of the respective companies listed below.

---

# Data Sources

Reports used during development:

- Apple Environmental Progress Report (2025, 2026)
- Google Environmental Report (2025, 2026)
- Microsoft Sustainability Report (2025, 2026)
- Reliance Sustainability Report (2025)
- Tata Sustainability Report (2025)

---

# Known Limitations

### Full Vector Index Rebuild

Uploading new PDF reports currently rebuilds the entire FAISS index instead of incrementally adding only the new documents.

---

### Chart & Table Extraction

Numeric values extracted from charts and tables may occasionally be less reliable than paragraph text because `pdfplumber` flattens chart content without preserving layout.

---

### Comparison Scope

The comparison engine is optimized for **cross-company comparisons** (e.g., Apple vs Google).

Year-over-year comparisons within the same company (e.g., Microsoft 2025 vs Microsoft 2026) currently use the standard retrieval pipeline instead of a dedicated comparison workflow.

---

### PDF-only Support

Currently, only PDF documents are supported.

Other formats such as Word documents, Excel sheets, or CSV files are not yet supported.

---

# Future Enhancements

- [ ] Incremental vector index updates
- [ ] Dedicated year-over-year comparison mode
- [ ] Retrieval evaluation using **RAGAS**
- [ ] Improved chart and table extraction pipeline
- [ ] Support for additional document formats (DOCX, CSV, XLSX)
- [ ] Conversation memory for multi-turn Q&A

---

# License

This project is licensed under the **MIT License**.
