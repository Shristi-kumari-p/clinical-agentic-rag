"""Central configuration. Any value below can be overridden via a .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ---- Paths ---------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Qdrant Vector Database -----------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "clinical_evidence"
UPLOAD_BATCH_SIZE = 256

# ---- Sources -------------------------------------------------------------
SOURCE_CLINICAL = "ClinicalTrials.gov"
SOURCE_FDA = "FDA"
SOURCE_PUBMED = "PubMed"
SOURCE_DAILYMED = "DailyMed"

ID_PREFIX = {
    SOURCE_FDA: "fda",
    SOURCE_CLINICAL: "ct",
    SOURCE_PUBMED: "pm",
    SOURCE_DAILYMED: "dm"
}
ALL_SOURCES = list(ID_PREFIX.keys())

# ---- APIs ----------------------------------------------------------------
CT_URL = "https://clinicaltrials.gov/api/v2/studies"
CT_PAGE_SIZE = 1000          # API maximum
N_CLINICAL_TRIALS = 10_000

FDA_URL = "https://api.fda.gov/drug/event.json"
FDA_PAGE_SIZE = 100
N_FDA = 100                  # openFDA allows skip values up to 25,000

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_TERM = "diabetes"
N_PUBMED = 100
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")

# ---- Chunking / Embeddings / Reranking -----------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_DOC_CHARS = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 768
EMBED_BATCH_SIZE = 32
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
N_CANDIDATES = 20            # Candidates fetched from Qdrant before reranking
TOP_K = 5                    # Kept after reranking

# ---- LLM -----------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
LLM_REPO_ID = os.getenv("LLM_REPO_ID", "deepseek-ai/DeepSeek-V3-0324")
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.2
