"""
Content guardrails — injectable input/output screening.

The default policy lives in `DEFAULT_SAFETY`; callers can pass a custom
`SafetyConfig` to substitute different blocklists or thresholds without
monkey-patching this module.
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SafetyConfig:
    """Policy bundle used by the validators below."""
    min_topic_length:  int
    max_topic_length:  int
    min_output_length: int
    blocked_input_words:  tuple[str, ...]
    sensitive_patterns:   tuple[str, ...]
    blocked_output_terms: tuple[str, ...]


# ── Default policy ────────────────────────────────────────────────────────────
DEFAULT_SAFETY = SafetyConfig(
    min_topic_length=3,
    max_topic_length=400,
    min_output_length=200,
    blocked_input_words=(
        "kill", "murder", "blood", "gore", "dead", "death", "die", "dying",
        "sex", "sexual", "porn", "naked", "nude",
        "drug", "cocaine", "heroin", "meth", "weed", "alcohol", "drunk", "beer", "wine",
        "gun", "shoot", "stab", "bomb", "weapon", "knife attack",
        "terror", "terrorist", "suicide", "self-harm", "abuse",
        "racist", "racism", "hate speech", "assault",
        "gambling", "casino",
    ),
    sensitive_patterns=(
        r"\bdead\b(?!\s+(?:tree|end|leaf|heat|calm|ringer|pan|lock))",
        r"\bkill(?:ing)?\b(?!\s+(?:time|it|switch|joy))",
    ),
    blocked_output_terms=("explicit", "adult content", "sexual", "pornograph"),
)


# ── Validators ────────────────────────────────────────────────────────────────

def check_input(topic: str, config: SafetyConfig = DEFAULT_SAFETY) -> tuple[bool, str]:
    """Validate a story topic. Returns (is_safe, error_message)."""
    topic = topic.strip()

    if not topic:
        return False, "Please enter a story topic!"

    if len(topic) < config.min_topic_length:
        return False, "Topic is too short — give me at least a few words to work with."

    if len(topic) > config.max_topic_length:
        return False, f"Topic is too long. Keep it under {config.max_topic_length} characters!"

    lower = topic.lower()

    for word in config.blocked_input_words:
        if word in lower:
            return False, (
                "Hmm, that topic isn't quite right for our young readers. "
                "Try something fun and adventurous instead!"
            )

    for pattern in config.sensitive_patterns:
        if re.search(pattern, lower):
            return False, (
                "That topic might not be suitable for kids. "
                "How about something magical or adventurous?"
            )

    return True, ""


def check_output(text: str, config: SafetyConfig = DEFAULT_SAFETY) -> tuple[bool, str]:
    """Validate generated story text. Returns (is_safe, error_message)."""
    if len(text) < config.min_output_length:
        return False, "The story was too short. Please try generating again."

    for term in config.blocked_output_terms:
        if term in text.lower():
            return False, "Story content was flagged as inappropriate. Please try again."

    return True, ""
