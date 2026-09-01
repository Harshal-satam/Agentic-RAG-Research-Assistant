# 🧠 Agentic RAG Research Assistant

> **An intelligent research assistant that combines Retrieval-Augmented Generation (RAG), semantic search, and agentic planning to transform academic research papers into grounded, context-aware answers.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://github.com/facebookresearch/faiss)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

---

## 📌 Overview

The **Agentic RAG Research Assistant** is an AI-powered research system designed to help users interact with academic papers through **semantic retrieval, intelligent planning, and grounded question answering**.

Instead of relying solely on an LLM's pretrained knowledge, the system retrieves relevant information from a collection of research documents and uses that context to generate more reliable responses.

The project extends traditional RAG by introducing an **agentic planning layer**, allowing the system to determine how a research query should be processed before generating an answer.

### Core Objective

> **Query → Plan → Retrieve → Augment → Generate → Answer**

This architecture demonstrates how modern AI systems can combine **LLMs, vector databases, semantic embeddings, and agentic workflows** into a practical research application.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │       User Query     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Planner Agent     │
                    │ Query Understanding  │
                    │ & Task Decomposition │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Query Processing   │
                    │  Semantic Embedding  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FAISS Vector Store │
                    │   Similarity Search  │
                    └──────────┬───────────┘
                               │
                         Top-K Documents
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Retrieved Context   │
                    │  Research Evidence   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    LLM Generation    │
                    │ Context-Grounded QA  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Final Response    │
                    │ + Retrieved Evidence │
                    └──────────────────────┘
```

---

# 🚀 Key Features

### 🔎 Semantic Research Retrieval

Research papers are converted into vector representations using transformer-based embeddings.

The system performs semantic similarity search rather than relying only on keyword matching.

---

### 🤖 Agentic Query Planning

A planning component analyzes incoming questions and determines an appropriate processing strategy.

This enables the system to move beyond a simple:

```text
Question → Search → Answer
```

pipeline toward:

```text
Question
   ↓
Understand
   ↓
Plan
   ↓
Retrieve
   ↓
Evaluate Context
   ↓
Generate
```

---

### 📚 PDF Research Paper Processing

The system can process academic PDF documents and transform their contents into searchable chunks.

The pipeline includes:

* PDF ingestion
* Text extraction
* Document chunking
* Metadata preservation
* Embedding generation
* Vector indexing

---

### 🧩 Vector Database

**FAISS (Facebook AI Similarity Search)** is used to efficiently perform nearest-neighbor similarity searches over document embeddings.

This allows the assistant to identify the most semantically relevant sections of research papers.

---

### 🧠 Context-Grounded Generation

Retrieved documents are supplied as contextual evidence to the language model.

This reduces dependence on general pretrained knowledge and helps the system produce responses grounded in the indexed research material.

---

### 📊 Retrieval Evaluation

The project incorporates retrieval-quality evaluation using metrics such as:

* **Precision@K**
* **Recall@K**
* **F1@K**

These metrics help evaluate whether the retrieval component is returning relevant research content.

---

### 🖥️ Interactive Streamlit Interface

A lightweight Streamlit interface provides an accessible way to interact with the research assistant without directly executing the underlying pipeline.

---

# 🛠️ Technology Stack

| Category        | Technology                 |
| --------------- | -------------------------- |
| Programming     | Python                     |
| Development     | Jupyter Notebook           |
| LLM Framework   | LangChain                  |
| Agent Workflow  | LangGraph / Agent Pipeline |
| Embeddings      | Sentence Transformers      |
| Vector Database | FAISS                      |
| PDF Processing  | PyMuPDF                    |
| Frontend        | Streamlit                  |
| Data Processing | NumPy, Pandas              |
| Environment     | Python Virtual Environment |
| Version Control | Git & GitHub               |

---

# 📂 Project Structure

```text
Agentic-RAG-Research-Assistant/
│
├── app/
│   ├── app.py
│   └── rag_backend.py
│
├── data/
│   └── research_papers/
│       └── r1.pdf
│
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_pdf_processing.ipynb
│   └── 03_planner_agent.ipynb
│
├── vector_store/
│   └── r1_index/
│       ├── index.faiss
│       └── index.pkl
│
├── agent_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Harshal-satam/Agentic-RAG-Research-Assistant.git
cd Agentic-RAG-Research-Assistant
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file based on the provided example:

```bash
copy .env.example .env
```

Add the required API credentials inside `.env`.

Example:

```env
GOOGLE_API_KEY=your_api_key_here
```

> **Never commit API keys or secrets to GitHub.**

The `.gitignore` file is configured to prevent sensitive environment files from being committed.

---

# ▶️ Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The Streamlit interface will open in your browser.

---

# 🧪 Research Workflow

The project is organized into progressive notebooks so that each stage of the system can be understood independently.

### Notebook 01 — Environment Setup

Establishes the Python environment and verifies the required dependencies.

### Notebook 02 — PDF Processing

Demonstrates:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Index
```

### Notebook 03 — Planner Agent

Introduces the agentic component responsible for analyzing and planning research queries.

---

# 📈 Evaluation

The retrieval pipeline can be evaluated using Top-K retrieval metrics.

### Precision@K

Measures how many retrieved documents are actually relevant.

```text
Precision@K =
Relevant Retrieved Documents
────────────────────────────
Total Retrieved Documents
```

### Recall@K

Measures how many of the relevant documents were successfully retrieved.

```text
Recall@K =
Relevant Retrieved Documents
────────────────────────────
Total Relevant Documents
```

### F1@K

Provides a harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall
     ───────────────────────
       Precision + Recall
```

These metrics provide a quantitative view of retrieval quality rather than evaluating the system solely through subjective responses.

---

# 🧠 Why Agentic RAG?

Traditional RAG systems generally follow:

```text
User Query
     ↓
Retriever
     ↓
Context
     ↓
LLM
     ↓
Answer
```

This project introduces a planning layer:

```text
User Query
     ↓
Planner Agent
     ↓
Query Strategy
     ↓
Retriever
     ↓
Context
     ↓
LLM
     ↓
Grounded Answer
```

The additional reasoning layer provides a foundation for extending the system toward more advanced research workflows such as:

* Multi-document reasoning
* Query decomposition
* Iterative retrieval
* Evidence verification
* Research summarization
* Citation-aware generation
* Autonomous literature analysis

---

# 🔬 Potential Research Extensions

The current implementation provides a foundation for several advanced research directions.

### 1. Multi-Agent Research System

Introduce specialized agents:

```text
Planner Agent
      │
      ├── Retrieval Agent
      ├── Summarization Agent
      ├── Verification Agent
      └── Citation Agent
```

---

### 2. Adaptive Retrieval

Allow the system to dynamically determine whether additional retrieval is required based on the quality of the retrieved context.

---

### 3. Hybrid Search

Combine:

```text
Semantic Search
      +
Keyword Search
      +
Metadata Filtering
```

to improve retrieval robustness.

---

### 4. Reranking

Introduce a cross-encoder reranking stage after FAISS retrieval to improve Top-K relevance.

---

### 5. Research Knowledge Graph

Extract entities and relationships from papers to construct a research-oriented knowledge graph.

---

### 6. Citation-Aware Answers

Generate responses with direct references to the retrieved paper sections.

---

### 7. Hallucination Detection

Add an evidence-verification agent that evaluates whether generated claims are supported by retrieved context.

---

# 🎯 Skills Demonstrated

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Agentic AI architectures
* Large Language Model applications
* Semantic search
* Transformer embeddings
* Vector databases
* FAISS
* LangChain
* LangGraph
* PDF document processing
* Information retrieval evaluation
* Prompt engineering
* Streamlit application development
* Python-based AI pipelines
* Git/GitHub version control

---

# 💡 Use Cases

The architecture can be adapted for:

📚 **Academic Research**

Search and interact with collections of research papers.

🏦 **Financial Research**

Analyze financial reports, filings, and investment research.

⚖️ **Legal Research**

Retrieve relevant sections from large collections of legal documents.

🏥 **Scientific Literature**

Assist researchers in exploring scientific publications.

🏢 **Enterprise Knowledge Search**

Build internal question-answering systems over company documentation.

---

# 🔒 Security Considerations

* API keys are stored using environment variables.
* `.env` files are excluded through `.gitignore`.
* Sensitive credentials should never be committed to version control.
* Production deployments should additionally implement authentication, access control, logging, and secure secret management.

---

# 📌 Current Status

**Project Status: ✅ Completed**

The current implementation includes:

* [x] Environment setup
* [x] PDF ingestion
* [x] Text processing
* [x] Document chunking
* [x] Embedding generation
* [x] FAISS vector indexing
* [x] Semantic retrieval
* [x] Retrieval evaluation
* [x] Agentic planning
* [x] RAG backend
* [x] Streamlit interface
* [x] Git/GitHub integration

---

# 👨‍💻 Author

**Harshal Satam**

Computer Engineering | AI & Data Analytics Enthusiast

GitHub: **Harshal-satam**

---

# 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Acknowledgements

This project builds upon the open-source ecosystem surrounding:

* LangChain
* LangGraph
* FAISS
* Sentence Transformers
* PyMuPDF
* Streamlit
* Jupyter

---

> **Built as an exploration of Agentic AI, Retrieval-Augmented Generation, and intelligent research automation.**
