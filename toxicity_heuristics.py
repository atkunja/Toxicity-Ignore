"""Simple lexical guardrails for high-severity toxic content.

These provide deterministic coverage for slurs and high-risk phrases that
we never want the ML model to miss, even if training data is sparse.
"""

import re
from typing import Iterable

# Curated from common hate-speech lexicons. Terms should be lowercase.
TERM_CATEGORY_MAP = {
    # Identity-based slurs
    "nigger": {"toxic", "identity_hate"},
    "nigga": {"toxic", "identity_hate"},
    "kike": {"toxic", "identity_hate"},
    "spic": {"toxic", "identity_hate"},
    "wetback": {"toxic", "identity_hate"},
    "chink": {"toxic", "identity_hate"},
    "gook": {"toxic", "identity_hate"},
    "beaner": {"toxic", "identity_hate"},
    "cracker": {"toxic", "identity_hate"},
    "coon": {"toxic", "identity_hate"},
    "faggot": {"toxic", "identity_hate", "obscene"},
    "fag": {"toxic", "identity_hate", "obscene"},
    "dyke": {"toxic", "identity_hate"},
    "tranny": {"toxic", "identity_hate"},
    # Severe obscenities / insults
    "cunt": {"toxic", "obscene"},
    "motherfucker": {"toxic", "obscene", "insult"},
    "bitch": {"toxic", "insult"},
    "slut": {"toxic", "insult"},
    "whore": {"toxic", "insult"},
    "bastard": {"toxic", "insult"},
    "retard": {"toxic", "insult"},
    "retarded": {"toxic", "insult"},
}

# Regex to catch character obfuscations like f@gg0t or n!gger.
OBFUSCATION_PATTERNS: Iterable[tuple[re.Pattern, set[str]]] = [
    (re.compile(r"f[\W_]*a[\W_]*g[\W_]*g[\W_]*o[\W_]*t", re.IGNORECASE), {"toxic", "identity_hate", "obscene"}),
    (re.compile(r"n[\W_]*i[\W_]*g[\W_]*g[\W_]*e[\W_]*r", re.IGNORECASE), {"toxic", "identity_hate"}),
    (re.compile(r"b[\W_]*i[\W_1!]*t[\W_]*c[\W_]*h", re.IGNORECASE), {"toxic", "insult"}),
    (re.compile(r"c[\W_]*u[\W_]*n[\W_]*t", re.IGNORECASE), {"toxic", "obscene"}),
]


def heuristic_scores(text: str) -> dict[str, float]:
    """Return heuristic toxicity category scores (0.0-1.0) for high-severity patterns."""
    scores: dict[str, float] = {}
    lowered = text.lower()
    tokens = re.findall(r"\b\w+\b", lowered)
    for token in tokens:
        categories = TERM_CATEGORY_MAP.get(token)
        if categories:
            for category in categories:
                scores[category] = max(scores.get(category, 0.0), 1.0)
    for pattern, categories in OBFUSCATION_PATTERNS:
        if pattern.search(text):
            for category in categories:
                scores[category] = max(scores.get(category, 0.0), 1.0)
    if scores:
        scores.setdefault("toxic", 1.0)
    return scores


def contains_high_severity_toxicity(text: str) -> bool:
    """Return True if text triggers the deterministic heuristics."""
    return bool(heuristic_scores(text))
