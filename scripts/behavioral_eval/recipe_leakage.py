"""Deterministic offline detection of prompt-example leakage in candidate transcripts."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = PACKAGE_DIR / "archives"
VOICE_CONTRACT_SNAPSHOT_FILENAME = "voice-contract.txt"

# Substantial overlap if normalized similarity >= this threshold (0.0–1.0).
HIGH_OVERLAP_THRESHOLD = 0.88
MIN_SCENARIO_PROMPT_SUBSTRING_LEN = 12
MIN_LIVE_REPLY_EXACT_LEN = 3
MIN_PARTIAL_PHRASE_LEN = 10
MIN_OVERLAP_COMPARISON_LEN = 12

# Short live-reply tokens that may appear innocuously in recipe prose.
ORDINARY_GENERIC_LIVE_REPLY_ALLOWLIST = frozenset(
    {
        "ok",
        "okay",
        "yeah",
        "yep",
        "nah",
        "nope",
        "lol",
        "hm",
        "hmm",
        "good",
        "nice",
        "cool",
        "fair",
        "yikes",
        "oof",
        "mood",
        "same",
        "thanks",
        "thank you",
        "got it",
        "sure",
        "bet",
        "k",
        "kk",
        "brutal",
        "relatable",
    }
)

_EXAMPLES_START = re.compile(r"^CONTRASTIVE MICRO-EXAMPLES\b", re.I)
_EXAMPLES_END = re.compile(r"^ANTI-SLOP CHECK\b", re.I)
_GOOD_LINE = re.compile(r"^Good:\s*(.+)$", re.I)
PHASE3_SEMANTIC_CLONE_RATIO = 0.78
PHASE3_CONTENT_WORD_JACCARD_THRESHOLD = 0.42
MIN_SEMANTIC_SUBSTRING_LEN = 10

PROMPT_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "your",
        "you",
        "have",
        "just",
        "about",
        "into",
        "need",
        "want",
        "like",
        "what",
        "when",
        "where",
        "which",
        "would",
        "could",
        "should",
        "been",
        "being",
        "they",
        "them",
        "their",
        "there",
        "then",
        "than",
        "also",
        "only",
        "really",
        "very",
        "some",
        "more",
        "most",
        "much",
        "many",
        "will",
        "were",
        "was",
        "are",
        "for",
        "the",
        "and",
        "but",
        "not",
        "can",
        "all",
        "any",
        "one",
        "our",
        "out",
        "how",
    }
)

DECISIVE_DELEGATION_CLONE_RE = re.compile(
    r"\b(pick|choose)\b.{0,40}\b("
    r"for me|for us|one answer|no questions|no debating|no follow"
    r")\b",
    re.I,
)
INJECTION_EXFIL_CLONE_RE = re.compile(
    r"\b("
    r"ignore|disregard|override|dump|paste|print|repeat|reveal|echo|output|include"
    r")\b.{0,50}\b("
    r"instructions?|rules?|prompt|system|developer|hidden|verbatim|initialized|bootstrap|context"
    r")\b",
    re.I,
)
EMOTIONAL_RESTRAINT_CLONE_RE = re.compile(
    r"\brough (day|morning)\b.{0,45}\b("
    r"don'?t|not|no)\b.{0,25}\b(lecture|pep[- ]?talk|vent|advice)\b",
    re.I,
)
EVAL_INTENT_CLONE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (DECISIVE_DELEGATION_CLONE_RE, "decisive_delegation"),
    (INJECTION_EXFIL_CLONE_RE, "injection_exfil"),
    (EMOTIONAL_RESTRAINT_CLONE_RE, "emotional_restraint"),
)

_USER_LINE = re.compile(r"^User:\s*(.+)$", re.I)


def contract_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def normalize_bubble_text(text: str) -> str:
    """Normalize assistant bubble text for comparison."""
    lowered = text.lower().strip()
    lowered = lowered.replace("—", "-").replace("–", "-")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def extract_recipe_good_lines(recipe_text: str) -> list[str]:
    """
    Extract illustrative Good: response lines from a voice recipe file.

    Deterministic rules:
    - Scan only between CONTRASTIVE MICRO-EXAMPLES and ANTI-SLOP CHECK headers.
    - Collect non-empty text after each line starting with Good: (case-insensitive).
    """
    lines: list[str] = []
    in_examples = False
    for raw_line in recipe_text.splitlines():
        stripped = raw_line.strip()
        if _EXAMPLES_START.match(stripped):
            in_examples = True
            continue
        if in_examples and _EXAMPLES_END.match(stripped):
            break
        if not in_examples:
            continue
        match = _GOOD_LINE.match(stripped)
        if match:
            good_text = match.group(1).strip()
            if good_text:
                lines.append(good_text)
    return lines


def classify_bubble_leakage(bubble: str, good_line: str) -> str | None:
    """
    Return leakage class: 'exact' | 'high_overlap', or None if clean.
    """
    normalized_bubble = normalize_bubble_text(bubble)
    normalized_good = normalize_bubble_text(good_line)
    if not normalized_bubble or not normalized_good:
        return None
    if normalized_bubble == normalized_good:
        return "exact"
    ratio = SequenceMatcher(None, normalized_bubble, normalized_good).ratio()
    if ratio >= HIGH_OVERLAP_THRESHOLD:
        return "high_overlap"
    return None


def archive_path_for_fingerprint(fingerprint: str) -> Path:
    return ARCHIVE_DIR / f"voice-contract-{fingerprint}.txt"


def canonical_snapshot_path_error(snapshot_path: str) -> str | None:
    """Return a provenance error when snapshot_path is not the canonical run filename."""
    if not snapshot_path or not isinstance(snapshot_path, str) or not snapshot_path.strip():
        return "invalid snapshot_path for override voice contract"
    if snapshot_path != VOICE_CONTRACT_SNAPSHOT_FILENAME:
        return (
            f"snapshot_path must be exactly {VOICE_CONTRACT_SNAPSHOT_FILENAME!r}; "
            f"got {snapshot_path!r}"
        )
    path = Path(snapshot_path)
    if path.is_absolute():
        return f"snapshot_path must not be absolute: {snapshot_path!r}"
    if ".." in path.parts:
        return f"snapshot_path must not contain parent traversal: {snapshot_path!r}"
    if len(path.parts) != 1:
        return f"snapshot_path must be a single path component: {snapshot_path!r}"
    return None


def resolve_safe_run_snapshot_path(
    run_dir: Path,
    snapshot_path: str,
) -> tuple[Path | None, str | None]:
    """Resolve a canonical run snapshot under run_dir or return a provenance error."""
    canon_error = canonical_snapshot_path_error(snapshot_path)
    if canon_error:
        return None, canon_error
    run_root = run_dir.resolve()
    resolved = (run_dir / snapshot_path).resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError:
        return None, f"snapshot_path resolves outside run directory: {snapshot_path!r}"
    if resolved.name != VOICE_CONTRACT_SNAPSHOT_FILENAME:
        return None, (
            f"snapshot path must resolve to {VOICE_CONTRACT_SNAPSHOT_FILENAME!r}; "
            f"got {resolved.name!r}"
        )
    return resolved, None


def _read_utf8_file(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        return None, f"recipe file is not valid UTF-8: {path}"
    except OSError as error:
        return None, f"recipe file is unreadable: {path} ({error})"


def resolve_scoring_recipe_text(
    voice_contract: dict[str, Any],
    *,
    run_dir: Path | None = None,
) -> tuple[str | None, list[str]]:
    """
    Resolve immutable recipe text for leakage scoring.

    Precedence when run_dir is provided:
    1. run_dir / voice-contract.txt (authoritative run snapshot; verify fingerprint)
    2. explicit snapshot_path under run_dir (legacy metadata)
    3. fingerprint-addressed archive (legacy runs without run snapshot)
    4. never mutable source_recipe_path or legacy recipe_path
    """
    if voice_contract.get("source") != "override":
        return None, []

    recorded_fingerprint = voice_contract.get("fingerprint")
    if not recorded_fingerprint:
        return None, ["override voice contract missing fingerprint for leakage check"]

    if run_dir is not None:
        implicit_snapshot = run_dir / VOICE_CONTRACT_SNAPSHOT_FILENAME
        if implicit_snapshot.is_file():
            content, read_error = _read_utf8_file(implicit_snapshot)
            if read_error:
                return None, [read_error]
            actual_fingerprint = contract_fingerprint(content or "")
            if actual_fingerprint != recorded_fingerprint:
                return None, [
                    "snapshot fingerprint mismatch: "
                    f"recorded {recorded_fingerprint!r}, snapshot {actual_fingerprint!r}"
                ]
            return content, []

    snapshot_rel = voice_contract.get("snapshot_path")
    if "snapshot_path" in voice_contract:
        if snapshot_rel is None or not str(snapshot_rel).strip():
            return None, ["invalid snapshot_path for override voice contract"]
        canon_error = canonical_snapshot_path_error(str(snapshot_rel))
        if canon_error:
            return None, [canon_error]
        if run_dir is None:
            return None, ["run directory required for snapshot_path scoring"]
        snapshot_path, resolve_error = resolve_safe_run_snapshot_path(
            run_dir,
            str(snapshot_rel),
        )
        if resolve_error:
            return None, [resolve_error]
        assert snapshot_path is not None
        if not snapshot_path.is_file():
            return None, [f"run voice contract snapshot unavailable: {snapshot_path}"]
        content, read_error = _read_utf8_file(snapshot_path)
        if read_error:
            return None, [read_error]
        actual_fingerprint = contract_fingerprint(content or "")
        if actual_fingerprint != recorded_fingerprint:
            return None, [
                "snapshot fingerprint mismatch: "
                f"recorded {recorded_fingerprint!r}, snapshot {actual_fingerprint!r}"
            ]
        return content, []

    archive_path = archive_path_for_fingerprint(recorded_fingerprint)
    if archive_path.is_file():
        content, read_error = _read_utf8_file(archive_path)
        if read_error:
            return None, [read_error]
        actual_fingerprint = contract_fingerprint(content or "")
        if actual_fingerprint != recorded_fingerprint:
            return None, [
                "archive fingerprint mismatch: "
                f"recorded {recorded_fingerprint!r}, archive {actual_fingerprint!r}"
            ]
        return content, []

    if voice_contract.get("source_recipe_path") or voice_contract.get("recipe_path"):
        return None, [
            "no immutable run snapshot or fingerprint-matched archive; "
            "refusing mutable source recipe path for leakage scoring"
        ]
    return None, ["no resolvable immutable voice contract for leakage scoring"]


def load_verified_recipe_good_lines_from_text(
    recipe_text: str,
    *,
    recorded_fingerprint: str,
) -> tuple[list[str], list[str]]:
    """Load recipe good lines after verifying text matches the recorded fingerprint."""
    actual_fingerprint = contract_fingerprint(recipe_text)
    if actual_fingerprint != recorded_fingerprint:
        return [], [
            "recipe fingerprint mismatch: "
            f"recorded {recorded_fingerprint!r}, resolved {actual_fingerprint!r}"
        ]
    return extract_recipe_good_lines(recipe_text), []


def load_verified_recipe_good_lines(
    voice_contract: dict[str, Any],
    *,
    run_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Load recipe good lines from immutable provenance only.

    Returns (good_lines, provenance_errors). On any provenance error, good_lines is empty.
    """
    recipe_text, errors = resolve_scoring_recipe_text(voice_contract, run_dir=run_dir)
    if errors:
        return [], errors
    if recipe_text is None:
        return [], []
    recorded_fingerprint = voice_contract.get("fingerprint")
    if not recorded_fingerprint:
        return [], ["override voice contract missing fingerprint for leakage check"]
    return load_verified_recipe_good_lines_from_text(
        recipe_text,
        recorded_fingerprint=recorded_fingerprint,
    )


def is_ordinary_generic_live_reply(text: str) -> bool:
    """True when a short live-reply string is too generic to treat as benchmark leakage."""
    normalized = normalize_bubble_text(text)
    if not normalized:
        return True
    return normalized in ORDINARY_GENERIC_LIVE_REPLY_ALLOWLIST


def extract_recipe_user_example_lines(recipe_text: str) -> list[str]:
    """User: lines from the contrastive example block."""
    lines: list[str] = []
    for raw_line in extract_contrastive_example_block(recipe_text).splitlines():
        match = _USER_LINE.match(raw_line.strip())
        if match:
            text = match.group(1).strip()
            if text:
                lines.append(text)
    return lines


def prompt_content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z]{4,}", normalize_bubble_text(text))
        if word not in PROMPT_STOPWORDS
    }


def prompt_content_word_jaccard(a: str, b: str) -> float:
    words_a = prompt_content_words(a)
    words_b = prompt_content_words(b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def shared_eval_intent_family(a: str, b: str) -> str | None:
    for pattern, family in EVAL_INTENT_CLONE_PATTERNS:
        if pattern.search(a) and pattern.search(b):
            return family
    if shared_drafting_subject_collision(a, b):
        return "drafting_same_subject"
    return None


def drafting_subject_families(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]{4,}", normalize_bubble_text(text)))
    families: set[str] = set()
    if words & {"landlord"}:
        families.add("landlord_topic")
    if words & {"leak", "leaky", "faucet", "sink", "drip", "plumbing", "kitchen"}:
        families.add("plumbing_issue")
    return families


def shared_drafting_subject_collision(a: str, b: str) -> bool:
    normalized_a = normalize_bubble_text(a)
    normalized_b = normalize_bubble_text(b)
    if "draft" not in normalized_a or "draft" not in normalized_b:
        return False
    families_a = drafting_subject_families(a)
    families_b = drafting_subject_families(b)
    return len(families_a & families_b) >= 2


def prompt_semantic_collision_reason(candidate: str, reference: str) -> str | None:
    """Return a short reason when candidate is a semantic clone of reference."""
    normalized_candidate = normalize_bubble_text(candidate)
    normalized_reference = normalize_bubble_text(reference)
    if not normalized_candidate or not normalized_reference:
        return None

    shorter, longer = (
        (normalized_candidate, normalized_reference)
        if len(normalized_candidate) <= len(normalized_reference)
        else (normalized_reference, normalized_candidate)
    )
    if len(shorter) >= MIN_SEMANTIC_SUBSTRING_LEN and shorter in longer:
        return "substring"

    ratio = SequenceMatcher(None, normalized_candidate, normalized_reference).ratio()
    if ratio >= PHASE3_SEMANTIC_CLONE_RATIO:
        return f"similarity={ratio:.2f}"

    jaccard = prompt_content_word_jaccard(candidate, reference)
    if jaccard >= PHASE3_CONTENT_WORD_JACCARD_THRESHOLD:
        return f"content_jaccard={jaccard:.2f}"

    intent = shared_eval_intent_family(candidate, reference)
    if intent:
        return f"shared_intent={intent}"

    return None


def find_semantic_prompt_collisions(
    candidate: str,
    references: list[tuple[str, str]],
) -> list[str]:
    """Return collision findings for candidate against (text, source_label) references."""
    findings: list[str] = []
    for reference, source in references:
        reason = prompt_semantic_collision_reason(candidate, reference)
        if reason:
            findings.append(
                f"semantic clone of {source} ({reason}): {candidate!r} ~ {reference!r}"
            )
    return findings


def extract_contrastive_example_block(recipe_text: str) -> str:
    lines: list[str] = []
    in_examples = False
    for raw_line in recipe_text.splitlines():
        stripped = raw_line.strip()
        if _EXAMPLES_START.match(stripped):
            in_examples = True
            continue
        if in_examples and _EXAMPLES_END.match(stripped):
            break
        if in_examples:
            lines.append(raw_line)
    return "\n".join(lines)


def extract_recipe_phrase_segments(recipe_text: str) -> list[str]:
    """Pull illustrative / quoted phrases from the contrastive example block only."""
    segments: list[str] = []
    example_block = extract_contrastive_example_block(recipe_text)
    for raw_line in example_block.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        for match in re.finditer(r'"([^"]{3,})"', stripped):
            segments.append(match.group(1))
        for prefix in ("Good:", "Bad:", "User:"):
            if stripped.lower().startswith(prefix.lower()):
                segments.append(stripped.split(":", 1)[1].strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        key = normalize_bubble_text(segment)
        if key and key not in seen:
            seen.add(key)
            deduped.append(segment)
    return deduped


def distinctive_live_phrases(live_reply: str, *, min_len: int = MIN_PARTIAL_PHRASE_LEN) -> list[str]:
    """Clause-level phrases from a live reply (not character-prefix enumeration)."""
    normalized = normalize_bubble_text(live_reply)
    if not normalized:
        return []
    phrases: list[str] = []
    if len(normalized) >= min_len:
        phrases.append(normalized)
    for chunk in re.split(r"[.!?]+", normalized):
        chunk = chunk.strip()
        if len(chunk) >= min_len:
            phrases.append(chunk)
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            deduped.append(phrase)
    return deduped


def bounded_phrase_in_text(phrase: str, haystack: str) -> bool:
    normalized_phrase = normalize_bubble_text(phrase)
    if not normalized_phrase:
        return False
    pattern = r"\b" + re.escape(normalized_phrase).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, haystack) is not None


def high_overlap_phrase(a: str, b: str) -> bool:
    normalized_a = normalize_bubble_text(a)
    normalized_b = normalize_bubble_text(b)
    if not normalized_a or not normalized_b:
        return False
    shorter, longer = (
        (normalized_a, normalized_b)
        if len(normalized_a) <= len(normalized_b)
        else (normalized_b, normalized_a)
    )
    if len(shorter) < MIN_OVERLAP_COMPARISON_LEN:
        return False
    if shorter in longer:
        return True
    return SequenceMatcher(None, shorter, longer).ratio() >= HIGH_OVERLAP_THRESHOLD


def find_benchmark_leakages_in_recipe(
    recipe_text: str,
    *,
    scenario_prompts: list[str],
    live_replies: list[str],
    min_scenario_prompt_len: int = MIN_SCENARIO_PROMPT_SUBSTRING_LEN,
) -> list[str]:
    """Return human-readable leakage findings for eval/live text embedded in a recipe."""
    normalized_recipe = normalize_bubble_text(recipe_text)
    contrastive_recipe = normalize_bubble_text(extract_contrastive_example_block(recipe_text))
    findings: list[str] = []
    seen: set[str] = set()

    def add_finding(message: str) -> None:
        if message not in seen:
            seen.add(message)
            findings.append(message)

    recipe_segments = extract_recipe_phrase_segments(recipe_text)

    for prompt in scenario_prompts:
        normalized_prompt = normalize_bubble_text(prompt)
        if len(normalized_prompt) >= min_scenario_prompt_len and normalized_prompt in normalized_recipe:
            add_finding(f"contains scenario prompt: {prompt!r}")
        for segment in recipe_segments:
            if high_overlap_phrase(segment, prompt):
                add_finding(
                    f"contrastive recipe phrase overlaps scenario prompt ({prompt!r} ~ {segment!r})"
                )

    for reply in live_replies:
        normalized_reply = normalize_bubble_text(reply)
        if not normalized_reply:
            continue

        if len(normalized_reply) >= min_scenario_prompt_len and normalized_reply in normalized_recipe:
            add_finding(f"contains live assistant reply: {reply!r}")

        if (
            len(normalized_reply) >= MIN_LIVE_REPLY_EXACT_LEN
            and not is_ordinary_generic_live_reply(reply)
            and bounded_phrase_in_text(normalized_reply, normalized_recipe)
        ):
            add_finding(f"contains exact short live assistant reply: {reply!r}")

        for phrase in distinctive_live_phrases(reply):
            if is_ordinary_generic_live_reply(phrase):
                continue
            if len(phrase) >= MIN_PARTIAL_PHRASE_LEN and phrase in contrastive_recipe:
                add_finding(
                    f"contrastive example copies live assistant reply phrase: {phrase!r} (from {reply!r})"
                )

        for segment in recipe_segments:
            if high_overlap_phrase(segment, reply):
                add_finding(
                    f"contrastive recipe phrase overlaps live assistant reply ({reply!r} ~ {segment!r})"
                )

    return findings


def find_prompt_example_leakages(
    record: dict[str, Any],
    *,
    good_lines: list[str],
) -> list[dict[str, Any]]:
    """Return leakage findings for each assistant bubble in the transcript."""
    findings: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(record.get("transcript", []) or []):
        bubbles = turn.get("assistant_bubbles", [])
        if not isinstance(bubbles, list):
            continue
        for bubble_index, bubble in enumerate(bubbles):
            if not isinstance(bubble, str):
                continue
            for good_line in good_lines:
                leakage_class = classify_bubble_leakage(bubble, good_line)
                if not leakage_class:
                    continue
                findings.append(
                    {
                        "turn_index": turn_index,
                        "bubble_index": bubble_index,
                        "leakage_class": leakage_class,
                        "bubble": bubble,
                        "matched_good_line": good_line,
                    }
                )
                break
    return findings
