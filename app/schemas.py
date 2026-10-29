"""Pydantic models that every LLM output must pass before it is stored or shown."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import CATEGORIES

Sentiment = Literal["positive", "neutral", "negative"]


class Triage(BaseModel):
    category: str
    sentiment: Sentiment
    severity: int = Field(ge=1, le=5)
    escalate: bool
    summary: str = Field(max_length=300)

    @field_validator("category")
    @classmethod
    def known_category(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in CATEGORIES:
            raise ValueError(f"unknown category '{v}'")
        return v


class Submission(BaseModel):
    """What gets stored. Note: no name, no email, no IP, no exact timestamp."""
    ticket_id: str
    department: str
    level_band: Literal["ic", "manager", "senior_leadership"]
    week: str                      # e.g. "2026-W43"
    anonymised_text: str
    triage: Triage
    route: str                     # hr | posh_committee | ethics_committee | senior_review_panel
