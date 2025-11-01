"""Simple lexical guardrails for high-severity toxic content.

These provide deterministic coverage for slurs and high-risk phrases that
we never want the ML model to miss, even if training data is sparse.
"""

import re
from typing import Iterable

# Curated from common hate-speech lexicons. Terms should be lowercase.
HIGH_SEVERITY_TERMS = {
    "faggot",
    "fag",
    "dyke",
    "kike",
    "nigger",
    "nigga",
    "wetback",
    "spic",
    "tranny",
    "retard",
    "retarded",
    "cunt",
    "whore",
    "slut",
}

# Regex to catch character obfuscations like f@gg0t or n!gger.
OBFUSCATION_PATTERNS: Iterable[re.Pattern] = [
    re.compile(r"f[\W_]*a[\W_]*g[\W_]*g[\W_]*o[\W_]*t", re.IGNORECASE),
    re.compile(r"n[\W_]*i[\W_]*g[\W_]*g[\W_]*e[\W_]*r", re.IGNORECASE),
]


def contains_high_severity_toxicity(text: str) -> bool:
    """Return True if text contains a deterministic high-severity toxic signal."""
    lowered = text.lower()
    for term in HIGH_SEVERITY_TERMS:
        if f" {term} " in f" {lowered} ":
            return True
    return any(pattern.search(text) for pattern in OBFUSCATION_PATTERNS)
