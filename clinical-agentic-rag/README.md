# Clinical Agentic RAG

Answers clinical-research questions using evidence from three sources:
**ClinicalTrials.gov**, **openFDA** (drug adverse events) and **PubMed**.
A rule-based agent routes each question to the right source(s), ChromaDB retrieves
candidates, a cross-encoder reranks them, and an LLM writes an answer grounded
only in that evidence.

```
question -> route_query -> Chroma search (per source) -> cross-encoder rerank -> LLM answer
```

## Project layout

```
clinical-agentic-rag/
├── main.py                  # CLI: ingest | build-docs | index | status | ask | chat
├── clinical_rag/
│   ├── config.py            # all settings (paths, models, sizes)
│   ├── ingest.py            # API downloads
│   ├── documents.py         # document building, cleaning, chunking
│   ├── indexing.py          # BGE embeddings + ChromaDB
│   ├── retrieval.py         # router + retrieval + reranker
│   └── generation.py        # prompt, context builder, LLM chain
├── tests/                   # router + document tests
├── .vscode/                 # run/debug configs
├── .env.example
└── requirements.txt
```

## Setup in VS Code

1. Open the folder in VS Code (File > Open Folder).
2. Create and activate a virtual environment (Terminal > New Terminal):
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Select it: `Ctrl+Shift+P` > **Python: Select Interpreter** > `.venv`.
4. `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and paste your Hugging Face token into `HF_TOKEN`.

## Run the pipeline (in order)

```
python main.py ingest          # ~10,000 trials + 100 FDA reports + 100 PubMed articles
python main.py build-docs      # documents -> cleaned -> chunks
python main.py index           # embed and store in ChromaDB (slow on CPU for ~100k chunks)
python main.py status          # check chunk counts per collection
python main.py ask "What clinical trials are studying lung cancer?" --show-evidence
python main.py chat            # interactive
```

Sizes are set in `clinical_rag/config.py` (or via flags such as `--trials 500`).
For a quick test run: `python main.py ingest --trials 500` before `build-docs` and `index`.
The `.vscode/launch.json` file adds one-click Run and Debug entries for each step.

## Reusing an existing ChromaDB

If you already built the index in the notebook, set `CHROMA_PATH` in `.env` to that folder
and run `python main.py status`. If the counts look right you can skip `index`;
if anything looks off, re-run `python main.py index --reset`.

## Tests

`pip install pytest` (already in requirements) then `pytest`.
