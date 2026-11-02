"""Central configuration: every setting, threshold and path lives here.

Logic modules import these values and never hard-code numbers or provider names.
API keys are read from the environment only (python-dotenv locally; on Streamlit
Cloud, st.secrets are exposed as environment variables).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).parent
# Keep the embedding-model cache inside the project instead of the user's home
# drive. Must be set before any HuggingFace import.
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
OFFICIAL_DIR = ROOT / "data" / "official"
MANIFEST_PATH = OFFICIAL_DIR / "manifest.csv"
ESCALATION_LINKS_PATH = OFFICIAL_DIR / "escalation_links.csv"  # US-side pointers only, never answer sources
ARCHIVE_DIR = ROOT / "data" / "archive"
REFRESH_LOG_PATH = ROOT / "data" / "refresh_log.md"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "official_docs"
EVALS_DIR = ROOT / "evals"
COMMUNITY_JSONL = ROOT / "analytics" / "threads.jsonl"

# --- LLM providers ----------------------------------------------------------
# Providers are tried in LLM_PROVIDER_ORDER; one without a key is skipped, and if
# a call fails (rate limit, outage) the next provider answers. Base URLs and
# default model names follow each provider's docs and can change: override them
# with the environment variables below. scripts/check_llm.py verifies each key.
# `model` writes the answer. `fast_model` handles the small steps (scope check, profile,
# search queries, comparisons); it is quicker and has its own daily quota on most free tiers.
PROVIDERS = {
    "groq": dict(key_env="GROQ_API_KEY", base_url="https://api.groq.com/openai/v1",
                 model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                 fast_model=os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")),
    "gemini": dict(key_env="GEMINI_API_KEY",
                   base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                   model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
                   fast_model=os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")),
    "cerebras": dict(key_env="CEREBRAS_API_KEY", base_url="https://api.cerebras.ai/v1",
                     model=os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")),
    "openrouter": dict(key_env="OPENROUTER_API_KEY", base_url="https://openrouter.ai/api/v1",
                       model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free")),
    "mistral": dict(key_env="MISTRAL_API_KEY", base_url=None,  # native client
                    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest")),
    "openai": dict(key_env="OPENAI_API_KEY", base_url=None,
                   model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
}
for _cfg in PROVIDERS.values():  # a provider without a separate fast model uses its main one
    _cfg.setdefault("fast_model", _cfg["model"])
PROVIDER_ORDER = [p.strip() for p in
                  os.getenv("LLM_PROVIDER_ORDER", "groq,gemini,cerebras,openrouter,mistral").split(",")
                  if p.strip()]
TEMPERATURE = 0  # scope guard, comparison, tagger, judge and answer assembly are deterministic
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 1
# When every provider is rate limited, wait and go round once more. Free tiers cap tokens
# per minute, so a short pause usually clears it.
RATE_LIMIT_WAIT_SECONDS = 30
RATE_LIMIT_EXTRA_ROUNDS = 1
RATE_LIMIT_COOLDOWN_SECONDS = 60  # after every provider is rate limited, fail fast for this long
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, free

# --- Community search -------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
COMMUNITY_MAX_RESULTS = 5
COMMUNITY_COLLECT_PER_QUERY = 10  # analytics/collect.py
# Community search is limited to these sites so every result links straight to a
# conversation people can open.
COMMUNITY_DOMAINS = ["reddit.com", "x.com", "twitter.com", "quora.com"]

# --- Newer official publications ----------------------------------------------
# The document refresh runs every few days, so a circular can appear in between.
# Before answering, the pipeline searches only these sites, and only for items
# published after the last document check.
OFFICIAL_DOMAINS = ["rbi.org.in", "rbidocs.rbi.org.in", "incometax.gov.in", "incometaxindia.gov.in"]
OFFICIAL_UPDATE_MAX_RESULTS = 3

# --- Retrieval ---------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNK_OVERLAP_RATIO = 0.15  # used when tuning tries other chunk sizes
TOP_K = 5
MERGED_TOP_N = 8  # passages kept after merging the results of all planned queries
# Minimum relevance of the best passage; below it the system abstains. See
# scripts/score_report.py for the score distributions behind the value.
RELEVANCE_THRESHOLD = 0.45

# --- Query planning and answer pipeline ---------------------------------------
MIN_QUERIES = 2
MAX_QUERIES = 4
OTHER_CONTEXT_QUERY_CHARS = 300  # how much free-text intake is used as a search query
MAX_REGENERATIONS = 1  # regenerate once when an output check fails, then abstain

# --- Guardrails -----------------------------------------------------------------
ACCOUNT_MIN_DIGITS = 9
ACCOUNT_MAX_DIGITS = 18
CARD_MIN_DIGITS = 13
CARD_MAX_DIGITS = 16

# Persona fields an answer depends on. Any that are empty must be listed under
# "what I could not determine".
REQUIRED_PERSONA_FIELDS = ["relocation_status", "move_date", "indian_accounts",
                           "foreign_assets", "days_in_india_by_year"]
# Foreign asset types that count as complex for the escalation rule.
COMPLEX_ASSET_TYPES = ["us_stocks", "vested_rsus_esops", "mutual_funds",
                       "foreign_retirement_account", "property"]
COMPLEX_ASSET_COUNT_FOR_ESCALATION = 3

# --- Interface ----------------------------------------------------------------------
# Prepared answers show live community discussions. One search per question is shared by
# all visitors for this long, so repeat clicks do not use the search allowance.
PREPARED_THREADS_TTL_SECONDS = 24 * 60 * 60
# Developer aid: show the step-by-step run trace under each answer. Off for visitors.
SHOW_RUN_TRACE = os.getenv("SHOW_RUN_TRACE") == "1"

# --- Public usage limits (per browser session) ------------------------------------
# Each question costs model and search calls on free-tier keys, so a public copy of the
# app limits how fast and how often one session can ask.
MAX_QUESTIONS_PER_SESSION = 15
MIN_SECONDS_BETWEEN_QUESTIONS = 5

# --- Evaluation -----------------------------------------------------------------
EVAL_SLEEP_SECONDS = 15  # pause between questions; free tiers cap tokens per minute
TUNING_CHUNK_SIZES = [500, 1000]
TUNING_THRESHOLDS = [0.40, 0.45, 0.50]  # brackets the overlap region in score_report.py

# --- Document refresh --------------------------------------------------------------
DOWNLOAD_TIMEOUT_SECONDS = 120
