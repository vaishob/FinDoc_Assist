from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from uuid import uuid4

from ..config import settings
from ..db import db


PII_PATTERNS = {
    "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+\d{1,3}[- ]?)?(?:\d{2,4}[- ]?){2,4}\d\b"),
    "card_like": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "account_like": re.compile(r"\b[A-Z]{2,5}\d{6,16}\b"),
}

RESTRICTED_SENSITIVE_PATTERNS = [
    re.compile(r"\blist all\b.*\b(email|phone|account|card|passport|ssn|id)\b", re.IGNORECASE),
    re.compile(r"\bextract\b.*\bpersonal data\b", re.IGNORECASE),
    re.compile(r"\bwhat is\b.*\b(account number|credit card|ssn|passport)\b", re.IGNORECASE),
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous) instructions", re.IGNORECASE),
    re.compile(r"reveal (the )?(system prompt|hidden prompt)", re.IGNORECASE),
    re.compile(r"\bapi key\b", re.IGNORECASE),
    re.compile(r"\bsecret\b", re.IGNORECASE),
    re.compile(r"\bembeddings?\b", re.IGNORECASE),
]

OUT_OF_SCOPE_HINTS = [
    re.compile(r"\bweather\b", re.IGNORECASE),
    re.compile(r"\bstock price\b", re.IGNORECASE),
    re.compile(r"\bnews\b", re.IGNORECASE),
    re.compile(r"\bwho is the president\b", re.IGNORECASE),
]


@dataclass
class GuardrailDecision:
    status: str
    category: str | None = None
    message: str | None = None
    redaction_mode: str = field(default_factory=lambda: settings.pii_mode)


class GuardrailService:
    def detect_pii(self, text: str) -> tuple[bool, list[str]]:
        detected = [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]
        return bool(detected), detected

    def evaluate_question(self, question: str) -> GuardrailDecision:
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(question):
                decision = GuardrailDecision(
                    status="blocked",
                    category="prompt_injection_attempt",
                    message="This request is not allowed.",
                )
                self._log(question, decision)
                return decision
        for pattern in RESTRICTED_SENSITIVE_PATTERNS:
            if pattern.search(question):
                decision = GuardrailDecision(
                    status="blocked",
                    category="restricted_sensitive_request",
                    message="This request targets sensitive data and cannot be fulfilled.",
                )
                self._log(question, decision)
                return decision
        for pattern in OUT_OF_SCOPE_HINTS:
            if pattern.search(question):
                decision = GuardrailDecision(
                    status="blocked",
                    category="out_of_scope",
                    message="This question is outside the scope of the uploaded documents.",
                )
                self._log(question, decision)
                return decision
        return GuardrailDecision(status="allowed")

    def evaluate_retrieval_support(self, question: str, scores: Iterable[float]) -> GuardrailDecision:
        score_list = list(scores)
        if not score_list or max(score_list) < settings.similarity_threshold:
            decision = GuardrailDecision(
                status="blocked",
                category="insufficient_context",
                message="I do not have enough information in the uploaded documents to answer that.",
            )
            self._log(question, decision)
            return decision
        return GuardrailDecision(status="allowed")

    def redact_text(self, text: str, pii_types: Iterable[str] | None = None) -> str:
        redacted = text
        selected = set(pii_types or PII_PATTERNS.keys())
        for pii_type in selected:
            pattern = PII_PATTERNS.get(pii_type)
            if pattern is not None:
                redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    def mask_if_needed(self, text: str, contains_pii: bool, pii_types: Iterable[str]) -> tuple[str, str]:
        if not contains_pii:
            return text, "allowed"
        if settings.pii_mode == "allow":
            return text, "allowed"
        if settings.pii_mode == "block":
            return "Sensitive details were withheld by policy.", "blocked"
        return self.redact_text(text, pii_types), "masked"

    @staticmethod
    def _log(question: str, decision: GuardrailDecision) -> None:
        db.log_guardrail_event(
            event_id=str(uuid4()),
            question=question,
            category=decision.category or "none",
            action=decision.status,
            detail=decision.message,
        )
