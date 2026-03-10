# 🤖 AI Learning Series — From NLP to LLM Agents in 20 Days

> A structured, hands-on journey through modern AI — from classical NLP to production-ready RAG Agents.
> Every day is a self-contained Jupyter notebook: theory + working code + exercises.

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/🦜-LangChain-green)](https://langchain.com)

---

## 📖 About This Series

This repository documents a 20-day deep-dive into AI engineering — built to go from zero to building
production-quality LLM applications. The series is split into two arcs:

| Arc | Days | Focus |
|-----|------|-------|
| **NLP Foundations** | 1 – 10 | Classical NLP, embeddings, Transformers, HuggingFace |
| **LLMs & Agents** | 11 – 20 | LLMs, RAG, LangChain, Agents, Capstone Project |

Every notebook runs without an API key (local models + mock fallbacks) and includes deep explanations,
visualisations, reusable utility classes, and self-check questions.

---

## 🗂️ Curriculum

### Arc 1 — NLP Foundations (Days 1–10)

| Day | Notebook | Topics Covered |
|-----|----------|----------------|
| 1 | `Day-1(Basics of NLP).ipynb` | Tokenisation, stemming, lemmatisation, POS tagging, NER, spaCy pipeline |
| 2 | `Day-2(NLP-Foundational)-Part-1/2.ipynb` | TF-IDF, Bag of Words, n-grams, text preprocessing, feature engineering |
| 3 | `Day-3(Embeddings and classification).ipynb` | Word2Vec, GloVe, dense embeddings, cosine similarity, text classification |
| 4 | `Day-4(Spam Classification project).ipynb` | **Project** — End-to-end spam classifier: preprocessing → features → model → evaluation |
| 5 | `Day-5(Projects on NLP Foundational topics).ipynb` | **Projects** — Sentiment analysis, NER pipeline, text similarity system |
| 6 | `Day-6(Transformer Architecture).ipynb` | Attention mechanism, multi-head attention, positional encoding, encoder-decoder |
| 7 | `Day-7(Hugging face)-Part-1.ipynb` | HuggingFace Hub, Pipeline API, tokenisers (BPE/WordPiece), AutoModel |
| 8 | `Day-8(Hugging face)-Part-2.ipynb` | Fine-tuning BERT, Trainer API, LoRA mathematics, PEFT library |
| 9–10 | `Day-9,10(Interview focused questions).ipynb` | 50+ interview Q&A: NLP, Transformers, HuggingFace, model evaluation |

---

### Arc 2 — LLMs & Agents (Days 11–20)

| Day | Notebook | Topics Covered |
|-----|----------|----------------|
| 11 | `Day-11(LLM_Fundamentals).ipynb` | Pretraining, SFT, RLHF, tokens, context windows, temperature, API integration |
| 12 | `Day-12(Prompt_Engineering).ipynb` | Zero-shot, few-shot, Chain-of-Thought, self-consistency, JSON extraction, failure modes |
| 13 | `Day-13(Embeddings Semantic Search).ipynb` | Sentence embeddings, cosine similarity, FAISS, semantic vs keyword search |
| 14 | `Day-14(RAG Part1 Retrieval).ipynb` | Chunking strategies, VectorStore, metadata filtering, retrieval evaluation, re-ranking |
| 15 | `Day-15(RAG Part2 Generation).ipynb` | RAG prompt, source citation, multi-turn Q&A, HyDE, query expansion, LLM-as-Judge eval |
| 16 | `Day-16(LangChain Foundations).ipynb` | LCEL pipes, PromptTemplate, memory types, LangChain RAG, JsonOutputParser |
| 17 | `Day-17(Agents Tool Use).ipynb` | ReAct framework, `@tool` decorator, AgentExecutor, function calling, custom ReAct loop |
| 18 | `Day-18(RAG Agent Capstone).ipynb` | **Capstone** — Full RAG Agent: vector store + 5 tools + memory + auto-evaluation + session report |
| 19–20 | `Day-19,20(Interview Preparation).ipynb` | 60+ interview Q&A: LLMs, RAG, agents, system design, evaluation metrics |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Srinivasareddyseelam/AI_Learning_Series.git
cd AI_Learning_Series
```

### 2. Install dependencies

```bash
pip install torch transformers sentence-transformers datasets peft \
            faiss-cpu langchain langchain-openai langchain-anthropic \
            langchain-community openai anthropic tiktoken spacy \
            scikit-learn matplotlib numpy
```

### 3. (Optional) Set API keys for live LLM calls

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

> **No API key?** Every notebook has graceful fallbacks to local models or mock responses.
> You can read and run all code cells without spending a single token.

### 4. Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

---

## 🏗️ What You Will Build

By the end of this series you will have built the following systems from scratch:

```
✅  Spam classifier (TF-IDF + Naive Bayes / Logistic Regression)
✅  Semantic similarity engine (SentenceTransformer + cosine similarity)
✅  HuggingFace fine-tuning pipeline (BERT + LoRA + Trainer API)
✅  Prompt engineering library (classify, extract, summarise utilities)
✅  Semantic search engine (FAISS + all-MiniLM-L6-v2)
✅  RAG pipeline — retrieval (VectorStore + chunker + evaluator)
✅  RAG pipeline — generation (prompt + citation + refusal + HyDE)
✅  LangChain application (LCEL chains + memory + structured output)
✅  ReAct Agent with tools (calculator, search, compare, learning path)
✅  RAG Agent Capstone (8 documents + 5 tools + memory + eval dashboard)
```

---

## 🧠 Key Concepts by Day

<details>
<summary><strong>Arc 1: NLP Foundations</strong></summary>

- **Text preprocessing** — tokenisation, normalisation, stop-word removal
- **Feature engineering** — TF-IDF, BoW, n-gram language models
- **Word embeddings** — Word2Vec skip-gram, GloVe, dense vs sparse vectors
- **Transformer attention** — scaled dot-product, multi-head, positional encoding
- **HuggingFace ecosystem** — Pipeline API, tokenisers, AutoModel, datasets library
- **Fine-tuning** — Trainer API, LoRA mathematics, parameter-efficient training

</details>

<details>
<summary><strong>Arc 2: LLMs & Agents</strong></summary>

- **LLM training pipeline** — pretraining → SFT → RLHF → PPO
- **Prompt engineering** — CoT, self-consistency, JSON extraction, failure modes
- **Embeddings for retrieval** — sentence embeddings, FAISS, cosine similarity
- **RAG architecture** — chunking, vector stores, retrieval evaluation, re-ranking
- **LangChain** — LCEL, memory, chains, structured output
- **Agents** — ReAct loop, tool design, function calling, AgentExecutor
- **Evaluation** — faithfulness, relevance, context recall, LLM-as-Judge

</details>

---

## 📦 Tech Stack

| Category | Libraries |
|----------|-----------|
| NLP | `spaCy`, `NLTK`, `HuggingFace Transformers`, `datasets` |
| Embeddings | `sentence-transformers`, `FAISS` |
| Fine-tuning | `PEFT`, `LoRA`, `Trainer API` |
| LLM APIs | `openai`, `anthropic` |
| Frameworks | `LangChain`, `LCEL` |
| ML utilities | `scikit-learn`, `NumPy`, `Matplotlib` |
| Notebooks | `Jupyter`, `ipywidgets` |

---

## 📁 Repository Structure

```
AI_Learning_Series/
│
├── Day-1(Basics of NLP).ipynb
├── Day-2(NLP-Foundational)-Part-1.ipynb
├── Day-2(NLP-Foundational)-Part-2.ipynb
├── Day-3(Embeddings and classification).ipynb
├── Day-4(Spam Classification project).ipynb
├── Day-5(Projects on NLP Foundational topics).ipynb
├── Day-6(Transformer Architecture).ipynb
├── Day-7(Hugging face)-Part-1.ipynb
├── Day-8(Hugging face)-Part-2.ipynb
├── Day-9,10(Interview focused questions).ipynb
│
├── Day-11(LLM_Fundamentals).ipynb
├── Day-12(Prompt_Engineering).ipynb
├── Day-13(Embeddings Semantic Search).ipynb
├── Day-14(RAG Part1 Retrieval).ipynb
├── Day-15(RAG Part2 Generation).ipynb
├── Day-16(LangChain Foundations).ipynb
├── Day-17(Agents Tool Use).ipynb
├── Day-18(RAG Agent Capstone).ipynb
└── Day-19,20(Interview Preparation).ipynb
```

---

## 💡 How to Use This Series

**Learning path (recommended):**
- Follow days sequentially — each notebook builds on the previous
- Run every code cell before moving on
- Answer the self-check questions at the end of each day
- The project days (4, 5, 18) integrate multiple prior concepts — treat them as checkpoints

**Reference use:**
- Jump directly to any topic — each notebook is self-contained
- Days 9–10 and 19–20 are interview prep decks — review before interviews
- The `VectorStore`, `RAGPipeline`, and `AgentExecutor` patterns from Days 13–18 are production-ready templates

---

## 🎯 Who This Is For

- **ML engineers** transitioning into LLM / GenAI work
- **Data scientists** wanting structured exposure to modern NLP
- **Developers** building their first RAG or agent application
- **Anyone** preparing for AI/NLP engineering interviews

**Prerequisites:** Python proficiency, basic familiarity with machine learning concepts.
No prior NLP or LLM experience required.

---

## 📝 Author

**Srinivasareddy Seelam**
- GitHub: [@Srinivasareddyseelam](https://github.com/Srinivasareddyseelam)

---

## ⭐ If this series helped you

Give the repo a star — it helps others discover it and keeps the motivation going!

---

*Built with curiosity, one notebook at a time.*
