# AIActChecker

AI compliance assistant for the EU AI Act — built with RAG + LangGraph + Google Gemini.

## Overview

AIActChecker helps organizations understand and comply with the EU AI Act through four interactive modes:

- **Classify** — Determine if an AI system is high-risk, limited-risk, or minimal-risk under the EU AI Act
- **Obligations** — List the specific regulatory obligations that apply to a given AI system
- **Gap Analysis** — Compare current practices against EU AI Act requirements and identify gaps
- **Cross-Standard Map** — Map EU AI Act requirements to ISO/IEC 42001, NIST AI RMF, and GDPR

Built on top of academic research from my MSc capstone at IABG mbH (Business Analytics and Data Science, EU Business School Munich, 2026, graded 93/100), which adapted AI Governance frameworks to the EU AI Act for high-risk AI systems.

## Architecture

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Orchestration**: LangGraph (multi-agent workflows)
- **Vector Store**: ChromaDB
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Google text-embedding-004

## Documents Ingested

- EU AI Act (Regulation (EU) 2024/1689)
- GDPR (Regulation (EU) 2016/679)
- NIST AI Risk Management Framework
- ISO/IEC 42001 (AI Management Systems) — public summaries

## Quick Start

```bash
# 1. Clone
git clone https://github.com/sreeja1105/aiactchecker.git
cd aiactchecker

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env          # Windows
# cp .env.example .env          # Mac / Linux
# Then edit .env and paste your Gemini API key
# Get a free key at: https://aistudio.google.com/apikey

# 5. Download source documents
python scripts/download_docs.py

# 6. Ingest into ChromaDB
python scripts/ingest_docs.py

# 7. Test retrieval
python scripts/test_query.py
```

If step 7 prints answers with citations from the EU AI Act, the foundation is working.

## Status

🚧 Day 1: RAG foundation working — query EU AI Act, get cited answers.

Next phases:
- Day 2-3: Add ISO 42001, NIST RMF, GDPR documents
- Day 4-7: Build the four modes (Classify, Obligations, Gap, Cross-Map) as LangGraph agents
- Day 8-10: FastAPI endpoints
- Day 11-14: Streamlit UI with all modes

## License

MIT
