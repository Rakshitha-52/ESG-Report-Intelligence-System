# ESG Report Intelligence System

Imagine reading a 150-page sustainability report just to find one carbon emission target.

This project eliminates that problem.

ESG Report Intelligence System is an AI-powered assistant that reads ESG and Sustainability Reports, understands their content, and answers questions in plain English with supporting references from the original documents.

---

## What problem does it solve?

Companies publish large ESG reports containing information about:

- Carbon emissions
- Renewable energy usage
- Sustainability goals
- Waste management
- Social responsibility initiatives
- Governance policies

Finding specific information manually is time-consuming.

This system allows users to simply ask:

> "What is the company's carbon neutrality target?"

and receive an answer generated from the report itself.

---

## How it works

PDF Reports
      ↓
Text Extraction
      ↓
Document Chunking
      ↓
Vector Embeddings
      ↓
FAISS Vector Database
      ↓
Semantic Retrieval
      ↓
LLM Response Generation
      ↓
Answer + Source References

---

## Technologies Used

Python • LangChain • FAISS • Sentence Transformers • Streamlit • OpenAI/Gemini • PyPDF2 • PDFPlumber

---

## Sample Questions

- What are the renewable energy goals?
- How much did emissions reduce in 2023?
- What sustainability initiatives were introduced?
- Compare ESG targets across multiple reports.
- What is the company's net-zero roadmap?

---

## Why I Built This

I wanted to understand how modern AI systems combine:

- Natural Language Processing
- Vector Databases
- Large Language Models
- Retrieval-Augmented Generation (RAG)

instead of building another generic chatbot.

This project helped me learn how production-grade AI applications retrieve information from private documents and generate grounded responses.

---

## Highlights

✓ Multi-PDF document processing

✓ Semantic search using embeddings

✓ Retrieval-Augmented Generation (RAG)

✓ Source-aware responses

✓ Streamlit interactive interface

✓ Modular architecture for future scaling

---

## Running Locally

```bash
git clone <repo-url>

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py