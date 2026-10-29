"""Central settings. Reads Streamlit secrets first (cloud), then .env (local)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    try:  # Streamlit Cloud secrets, if running inside Streamlit
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


GEMINI_API_KEY = _get("GEMINI_API_KEY")
GROQ_API_KEY = _get("GROQ_API_KEY")
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = _get("GROQ_MODEL", "openai/gpt-oss-120b")

LLM_MOCK = _get("LLM_MOCK", "0") == "1"
K_ANONYMITY_MIN = int(_get("K_ANONYMITY_MIN", "5"))
MAX_FEEDBACK_CHARS = int(_get("MAX_FEEDBACK_CHARS", "3000"))

DATA_DIR = ROOT / "data"
DB_PATH = ROOT / "feedback.db"
CHROMA_DIR = ROOT / "chroma_store"

CATEGORIES = [
    "compensation", "manager_behaviour", "workload",
    "harassment", "culture", "growth", "safety", "fraud", "hr_conduct", "other",
]
CATEGORY_LABELS = {
    "compensation": "Compensation", "manager_behaviour": "Manager behaviour",
    "workload": "Workload", "harassment": "Harassment", "culture": "Culture",
    "growth": "Growth", "safety": "Safety", "fraud": "Fraud / ethics",
    "hr_conduct": "Complaint about HR", "other": "Other",
}
ROUTE_LABELS = {
    "hr": "HR Business Partner", "posh_committee": "POSH Internal Committee",
    "ethics_committee": "Ethics Committee", "senior_review_panel": "Senior Review Panel",
}
# Broader grouping: department -> division
DIVISIONS = {
    "Engineering": "Tech", "Product": "Tech",
    "Sales": "Commercial", "Marketing": "Commercial", "Customer Support": "Commercial",
    "Operations": "Corporate", "Finance": "Corporate", "Human Resources": "Corporate",
}
SENIOR_POOL = "Senior leadership (company-wide)"
# Response-time targets in working days (from the handbook)
ACK_DAYS = {"hr": 2, "posh_committee": 3, "ethics_committee": 3, "senior_review_panel": 3}
# Categories that bypass the normal chain (see routing rules in README)
ESCALATION_ROUTES = {
    "harassment": "posh_committee",
    "safety": "ethics_committee",
    "fraud": "ethics_committee",
    "hr_conduct": "senior_review_panel",
}
