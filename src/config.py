"""Central configuration for the tender compliance agent.

All values can be overridden through environment variables (see .env.example).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LLM ---
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
# The newest flash models have a very small free-tier DAILY cap (~20 requests
# per model); a lite fallback lets a full analysis finish at zero cost.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16000"))


def llm_provider() -> str:
    """'anthropic' or 'gemini' — explicit LLM_PROVIDER wins, else whichever
    provider has a key configured (Anthropic first)."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("anthropic", "gemini"):
        return explicit
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"


def active_model() -> str:
    """The model the selected provider will use (for display)."""
    return GEMINI_MODEL if llm_provider() == "gemini" else ANTHROPIC_MODEL

# --- Data locations ---
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", PROJECT_ROOT / "data" / "knowledge"))
CASES_DIR = Path(os.getenv("CASES_DIR", PROJECT_ROOT / "data" / "sample_cases"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "data" / "chroma"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# --- Vector store collections ---
KNOWLEDGE_COLLECTION = "procurement_knowledge"
BIDS_COLLECTION = "supplier_bids"

# --- Workflow bounds (stopping conditions) ---
# Maximum times the supervisor may route before the workflow is force-finalized.
MAX_SUPERVISOR_STEPS = int(os.getenv("MAX_SUPERVISOR_STEPS", "8"))
# Maximum tool-use turns inside a single specialist's ReAct loop.
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "12"))
# Default number of chunks returned by a retrieval call.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def api_key_present() -> bool:
    """True when a credential for the selected provider is configured.

    Re-reads .env so a key added while the app is running is picked up on
    the next rerun without restarting the process.
    """
    load_dotenv(override=False)
    if llm_provider() == "gemini":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
