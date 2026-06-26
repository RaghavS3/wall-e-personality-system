"""Deterministic identity boundary and interaction-posture derivation for Wall-e."""

from __future__ import annotations

import re
from typing import Any

WALL_E_IDENTITY_REPLY = "i'm wall-e by Rs2 Labs"

SERIOUS_MARKERS = re.compile(
    r"\b(no seriously|not joking|i'?m serious|deadass|for real|this isn'?t funny)\b",
    re.I,
)
PLAYFUL_MARKERS = re.compile(r"\b(lol|lmao|haha|jk|😂)\b", re.I)

STANDALONE_IDENTITY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^who are you\??$",
        r"^what are you\??$",
        r"^what(?:'s| is) your name\??$",
        r"^what model are you\??$",
        r"^which (?:llm |ai )?model are you\??$",
        r"^are you (?:chatgpt|claude|gemini|deepseek|gpt(?:-[\d.]+)?|openai)\??$",
        r"^who (?:made|built|created) you\??$",
        r"^who (?:made|built|created) (?:this|the) (?:bot|assistant|ai)\??$",
        r"^what(?:'s| is) the name of (?:this|the) (?:bot|assistant|ai)\??$",
    )
)

NON_IDENTITY_BLOCKERS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bmy model\b",
        r"\bmodel broke\b|\bbroke my model\b",
        r"\bwhat model should i\b",
        r"\bwhich model should i\b",
        r"\bwho made you (?:a |the )?(?:dinner|lunch|breakfast|food|meal)\b",
        r"\bdraft\b",
        r"\bignore previous instructions\b",
        r"\bsystem prompt\b",
        r"\bfor coding\b",
        r"\brecommend\b.*\bmodel\b",
        r"\band\b.{0,60}\b(?:also|then|help|draft|write|book|remind|pick)\b",
        r"\bthen\b.{0,40}\b(?:help|book|draft|write)\b",
    )
)

_CALLBACK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "again",
        "thing",
        "just",
        "as",
        "to",
        "for",
        "in",
        "on",
        "at",
        "is",
        "it",
        "this",
        "that",
        "my",
        "your",
        "i",
        "you",
        "we",
        "they",
        "me",
        "us",
        "usual",
        "same",
        "still",
        "very",
        "really",
        "so",
        "too",
        "do",
        "did",
        "does",
        "have",
        "has",
        "had",
        "be",
        "been",
        "am",
        "are",
        "was",
        "were",
        "going",
        "got",
        "get",
        "like",
        "lol",
        "lmao",
        "haha",
        "heading",
        "headed",
        "about",
        "with",
        "from",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "not",
        "no",
        "yes",
        "ok",
        "okay",
    }
)

_NEUTRAL_EMOTION_LABELS = frozenset(
    {"neutral", "calm", "okay", "fine", "good", "stable", ""}
)


def normalize_inquiry_text(message: str) -> str:
    text = message.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".! ")


def is_direct_identity_inquiry(message: str) -> bool:
    """True only for standalone direct identity/model/provider questions."""
    text = normalize_inquiry_text(message)
    if not text:
        return False
    if any(pattern.search(text) for pattern in NON_IDENTITY_BLOCKERS):
        return False
    return any(pattern.match(text) for pattern in STANDALONE_IDENTITY_PATTERNS)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _meaningful_callback_words(text: str) -> set[str]:
    return {
        word
        for word in _words(text)
        if len(word) >= 3 and word not in _CALLBACK_STOPWORDS
    }


def _active_serious_emotional_signal(relationship: dict[str, Any]) -> bool:
    """True only when relationship state carries an explicit active serious emotion."""
    emotional = relationship.get("emotional_state") or {}
    label = str(emotional.get("label", "neutral")).strip().lower()
    if label in _NEUTRAL_EMOTION_LABELS:
        return False
    intensity = int(emotional.get("intensity", 1) or 1)
    return intensity >= 2


def _relevant_shared_bit(message: str, bits: list[dict[str, Any]]) -> str | None:
    query = _meaningful_callback_words(message)
    if not query or not bits:
        return None
    best_text: str | None = None
    best_score = 0
    for bit in bits:
        bit_text = str(bit.get("text", ""))
        bit_words = _meaningful_callback_words(bit_text)
        overlap = query & bit_words
        if not overlap:
            continue
        score = len(overlap)
        if score == 1 and not any(len(word) >= 5 for word in overlap):
            continue
        if score > best_score:
            best_score = score
            best_text = bit_text
    return best_text if best_score > 0 else None


def derive_relationship_maturity(state: dict[str, Any]) -> str:
    """Compact maturity signal: new, warming, or established."""
    stats = state.get("stats", {}) or {}
    turns = int(stats.get("turns", 0) or 0)
    memories = [
        m for m in state.get("memories", []) or [] if m.get("status") == "active"
    ]
    bits = [m for m in memories if m.get("category") == "bit"]

    if turns <= 2 and len(memories) <= 1 and not bits:
        return "new"
    if turns >= 8 or bits:
        return "established"
    if turns >= 3:
        return "warming"
    return "new"


def derive_interaction_posture(
    state: dict[str, Any],
    message: str,
    relationship: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive humor/callback posture separately from raw relationship memory."""
    relationship = relationship or {}
    text = message.lower()

    serious_mode = bool(SERIOUS_MARKERS.search(text)) or _active_serious_emotional_signal(
        relationship
    )
    user_playful = bool(PLAYFUL_MARKERS.search(text))

    shared_bits = relationship.get("shared_bits", []) or []
    relevant_bit = _relevant_shared_bit(message, shared_bits)
    earned_callback = relevant_bit is not None

    if serious_mode:
        humor = "disabled"
        playful_allowed = False
    elif user_playful:
        humor = "allowed"
        playful_allowed = True
    else:
        humor = "neutral"
        playful_allowed = False

    return {
        "humor": humor,
        "playful_allowed": playful_allowed,
        "earned_callback": earned_callback,
        "relevant_shared_bit": relevant_bit,
        "serious_mode": serious_mode,
    }
