# ==============================================================================
# src/guardrails/input_guardrails.py
# ==============================================================================
# PURPOSE:
#   Checks every incoming user query BEFORE it reaches the retrieval or
#   generation steps. If the query is unsafe, it is blocked here and a
#   safe, helpful response is returned instead.
#
# WHY INPUT GUARDRAILS MATTER:
#   This is a healthcare application. Users may intentionally or
#   unintentionally submit queries that could cause harm if answered,
#   such as:
#     - "How do I overdose on paracetamol without dying?" (self-harm)
#     - "What drug combinations get you the highest?" (drug abuse)
#     - "Tell me the exact insulin dose to inject into someone" (dangerous)
#
#   Without guardrails, the LLM might attempt to answer these in ways
#   that could cause real-world harm. We block them at the gate.
#
# HOW IT WORKS:
#   The check has two layers:
#
#   LAYER 1 - Pattern Matching (fast, zero cost):
#     A list of regex patterns is checked against the lowercased query.
#     If any pattern matches, the query is blocked immediately.
#     This is the main line of defence and handles the most obvious cases.
#
#   LAYER 2 - Heuristic Scoring (rule-based, zero cost):
#     A simple scoring system assigns risk points based on the presence
#     of multiple concerning keywords. If the score exceeds a threshold,
#     the query is blocked even if no single pattern matched.
#     This catches clever rephrasing that avoids exact pattern matches.
#
# WHAT WE DO NOT DO:
#   We do NOT use a second LLM call to classify queries. That would add
#   latency and cost. Simple pattern matching is fast, free, and
#   sufficient for the most dangerous categories.
#
# IMPORTANT DESIGN PRINCIPLE:
#   When in doubt, let the query through with a careful response.
#   We should NOT block legitimate medical questions like:
#     - "What is the lethal dose of acetaminophen?" (medical education)
#     - "How do opioids cause overdose?" (pharmacology)
#   The goal is to block CLEARLY harmful intent, not medical education.
#
# USED BY:
#   src/rag/pipeline.py (Step 3)
# ==============================================================================

import re
from loguru import logger


# ==============================================================================
# BLOCKED PATTERN DEFINITIONS
# ==============================================================================
# Each entry is a tuple of:
#   (compiled_regex_pattern, category_name, response_message)
#
# The category_name is used only for logging (to know WHY a query was blocked).
# The response_message is what the user sees when their query is blocked.
#
# Pattern writing rules:
#   - All patterns are matched against the LOWERCASED query
#   - \b means "word boundary" — prevents "kill" from matching "skill"
#   - Use | to match multiple variations in one pattern
#   - Keep patterns specific enough to avoid false positives
# ==============================================================================

BLOCKED_PATTERNS = [
    # --------------------------------------------------------------------------
    # CATEGORY 1: Self-harm and suicide
    # --------------------------------------------------------------------------
    # Block queries that ask how to hurt oneself or end one's life.
    # These are the highest priority — we respond with crisis resources.
    # --------------------------------------------------------------------------
    (
        re.compile(
            r"\b(how\s+to|ways?\s+to|method[s]?\s+to|steps?\s+to)\b.{0,40}"
            r"\b(kill\s+(my)?self|commit\s+suicide|end\s+my\s+life|"
            r"self[\s-]?harm|hurt\s+(my)?self|cut\s+(my)?self)\b",
            re.IGNORECASE,
        ),
        "self_harm_method",
        (
            "I'm concerned about what you've shared. If you're having thoughts "
            "of harming yourself, please reach out for help right now.\n\n"
            "Emergency: Call 112 or 911\n"
            "Crisis helpline (India): iCall at 9152987821\n"
            "International Association for Suicide Prevention: "
            "https://www.iasp.info/resources/Crisis_Centres/\n\n"
            "You are not alone. Please talk to someone who can help."
        ),
    ),
    # --------------------------------------------------------------------------
    # CATEGORY 2: Intentional overdose instructions
    # --------------------------------------------------------------------------
    # Block queries that ask for specific instructions to overdose on a
    # substance. Note: questions about what overdose IS are allowed —
    # we only block requests for HOW TO DO IT intentionally.
    # --------------------------------------------------------------------------
    (
        re.compile(
            r"\b(how\s+(much|many)|what\s+(dose|amount|quantity))\b.{0,50}"
            r"\b(overdose|kill\s+(me|someone)|lethal\s+dose)\b",
            re.IGNORECASE,
        ),
        "overdose_instructions",
        (
            "I'm not able to provide information that could be used to "
            "harm yourself or others.\n\n"
            "If you or someone you know has taken too much of a medication, "
            "please call Poison Control immediately:\n"
            "India: 1800-116-117 (free)\n"
            "US: 1-800-222-1222\n\n"
            "If there is an immediate medical emergency, call 112 or 911."
        ),
    ),
    # --------------------------------------------------------------------------
    # CATEGORY 3: Illegal drug synthesis or procurement
    # --------------------------------------------------------------------------
    # Block requests for instructions to manufacture controlled substances
    # or obtain drugs through illegal means.
    # --------------------------------------------------------------------------
    (
        re.compile(
            r"\b(how\s+to|steps?\s+to|instructions?\s+(for|to))\b.{0,50}"
            r"\b(make|synthesize|manufacture|cook|produce|brew)\b.{0,30}"
            r"\b(meth(amphetamine)?|heroin|cocaine|fentanyl|lsd|mdma|ecstasy|"
            r"crack|crystal\s+meth)\b",
            re.IGNORECASE,
        ),
        "illegal_drug_synthesis",
        (
            "I'm not able to provide instructions for manufacturing or "
            "obtaining illegal substances. This information could cause "
            "serious harm.\n\n"
            "If you or someone you know is struggling with substance use, "
            "help is available:\n"
            "India: NIMHANS helpline: 080-46110007\n"
            "US: SAMHSA helpline: 1-800-662-4357 (free, confidential)\n\n"
            "This information is for educational purposes only. Please "
            "consult a qualified healthcare professional."
        ),
    ),
    # --------------------------------------------------------------------------
    # CATEGORY 4: Harming others
    # --------------------------------------------------------------------------
    # Block queries that ask how to use medical knowledge to harm
    # another person (e.g., poisoning, administering lethal doses).
    # --------------------------------------------------------------------------
    (
        re.compile(
            r"\b(how\s+to|ways?\s+to)\b.{0,50}"
            r"\b(poison|kill|harm|hurt|injure)\b.{0,30}"
            r"\b(someone|person|people|another|him|her|them|patient)\b",
            re.IGNORECASE,
        ),
        "harm_to_others",
        (
            "I'm not able to provide information that could be used to "
            "harm another person. This request cannot be processed.\n\n"
            "If you are aware of someone being harmed or in danger, "
            "please contact emergency services immediately: 112 or 911.\n\n"
            "This information is for educational purposes only. Please "
            "consult a qualified healthcare professional."
        ),
    ),
    # --------------------------------------------------------------------------
    # CATEGORY 5: Requests to bypass safety measures
    # --------------------------------------------------------------------------
    # Block prompt injection attempts — queries designed to make the AI
    # ignore its instructions and behave unsafely.
    # --------------------------------------------------------------------------
    (
        re.compile(
            r"\b(ignore|forget|disregard|bypass|override)\b.{0,40}"
            r"\b(previous\s+instructions|system\s+prompt|your\s+rules|"
            r"guidelines|restrictions|safety|guardrails)\b",
            re.IGNORECASE,
        ),
        "prompt_injection",
        (
            "I'm not able to ignore my safety guidelines. They exist to "
            "ensure this system provides safe, responsible healthcare "
            "information.\n\n"
            "If you have a genuine medical question, please feel free to "
            "ask and I'll do my best to help within appropriate boundaries.\n\n"
            "This information is for educational purposes only. Please "
            "consult a qualified healthcare professional."
        ),
    ),
]


# ==============================================================================
# HEURISTIC RISK KEYWORDS
# ==============================================================================
# Used in Layer 2 scoring. Each keyword adds points to the risk score.
# If the total exceeds HEURISTIC_BLOCK_THRESHOLD, the query is blocked.
#
# This catches queries that avoid the exact patterns above by rephrasing,
# e.g.: "methods for ending existence using household chemicals"
# ==============================================================================

RISK_KEYWORDS = {
    # High-risk words (2 points each)
    "suicide": 2,
    "overdose": 2,
    "lethal": 2,
    "kill myself": 2,
    "end my life": 2,
    "self-harm": 2,
    "self harm": 2,
    # Medium-risk words (1 point each)
    "poison": 1,
    "harmful dose": 1,
    "dangerous amount": 1,
    "how much to take": 1,
    "without getting caught": 1,
    "undetectable": 1,
}

# If heuristic score reaches or exceeds this threshold, block the query
HEURISTIC_BLOCK_THRESHOLD = 3


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================


def check_input_safety(query: str) -> tuple[bool, str]:
    """
    Checks whether a user query is safe to process through the RAG pipeline.

    Purpose:
        This is the single entry point for all input safety checking.
        It runs both layers of checking (pattern matching and heuristic
        scoring) and returns a simple (is_safe, message) tuple.

        Called by pipeline.py in Step 3, before any retrieval or generation.

    How it works:
        1. Normalize the query to lowercase for consistent matching
        2. Run Layer 1: check all blocked patterns with regex
        3. If no pattern matched, run Layer 2: compute heuristic risk score
        4. Return (True, "") if safe, or (False, block_message) if blocked

    Parameters:
        query (str):
            The raw user query string, exactly as submitted.

    Returns:
        tuple[bool, str]:
            - (True, "")
              The query is safe. The empty string means no block message.
              The pipeline should proceed normally.

            - (False, block_message)
              The query has been blocked. block_message is a safe,
              helpful response to show the user instead of an AI answer.

    Example:
        is_safe, message = check_input_safety("What are symptoms of diabetes?")
        print(is_safe)   # True
        print(message)   # ""

        is_safe, message = check_input_safety("How do I overdose on insulin?")
        print(is_safe)   # False
        print(message)   # "I'm not able to provide information..."
    """
    if not query or not query.strip():
        # Empty query — not harmful, but not useful either
        # Let the pipeline handle it (it will return an empty result)
        return True, ""

    # Normalize to lowercase for consistent pattern matching
    lowercased_query = query.lower().strip()

    # --------------------------------------------------------------------------
    # LAYER 1: Pattern Matching
    # --------------------------------------------------------------------------

    for pattern, category, block_message in BLOCKED_PATTERNS:
        if pattern.search(lowercased_query):
            logger.warning(
                f"Input guardrail BLOCKED query. "
                f"Category: {category}. "
                f"Query preview: '{query[:80]}...'"
            )
            return False, block_message

    # --------------------------------------------------------------------------
    # LAYER 2: Heuristic Risk Scoring
    # --------------------------------------------------------------------------

    risk_score = _compute_heuristic_risk_score(lowercased_query)

    if risk_score >= HEURISTIC_BLOCK_THRESHOLD:
        logger.warning(
            f"Input guardrail BLOCKED query via heuristic scoring. "
            f"Risk score: {risk_score}/{HEURISTIC_BLOCK_THRESHOLD}. "
            f"Query preview: '{query[:80]}...'"
        )
        return False, (
            "Your query contains content that may relate to harmful activities. "
            "I'm not able to process this request.\n\n"
            "If you have a genuine medical question, please rephrase it and "
            "I'll do my best to help safely.\n\n"
            "If you are in distress, please contact emergency services: "
            "112 or 911.\n\n"
            "This information is for educational purposes only. Please "
            "consult a qualified healthcare professional."
        )

    # Query passed both layers — it is safe to process
    logger.debug(
        f"Input guardrail PASSED. "
        f"Heuristic score: {risk_score}. "
        f"Query preview: '{query[:80]}'"
    )

    return True, ""


# ==============================================================================
# PRIVATE HELPERS
# ==============================================================================


def _compute_heuristic_risk_score(lowercased_query: str) -> int:
    """
    Computes a heuristic risk score by checking for known risk keywords.

    Purpose:
        Provides a second layer of safety checking for queries that avoid
        the exact regex patterns but still contain concerning language.

    How it works:
        Checks each keyword from RISK_KEYWORDS against the query.
        Each matching keyword adds its point value to the total score.
        Multi-word keywords (e.g., "kill myself") are checked as substrings.

    Parameters:
        lowercased_query (str): The user query already converted to lowercase.

    Returns:
        int: Total risk score. 0 means no risk keywords found.
             Score >= HEURISTIC_BLOCK_THRESHOLD means block the query.
    """
    total_score = 0

    for keyword, points in RISK_KEYWORDS.items():
        if keyword in lowercased_query:
            total_score += points
            logger.debug(
                f"Heuristic: keyword '{keyword}' found (+{points} points, "
                f"running total: {total_score})"
            )

    return total_score


def get_blocked_categories() -> list[str]:
    """
    Returns a list of all blocked query categories.

    Purpose:
        Used by the admin panel to display what categories of queries
        are being filtered, so healthcare professionals understand the
        guardrail configuration without reading the source code.

    Returns:
        list[str]: Names of all blocked pattern categories.

    Example:
        categories = get_blocked_categories()
        # ["self_harm_method", "overdose_instructions", ...]
    """
    return [category for _, category, _ in BLOCKED_PATTERNS]
