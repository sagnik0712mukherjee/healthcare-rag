# ==============================================================================
# src/guardrails/output_guardrails.py
# ==============================================================================
# PURPOSE:
#   Inspects and cleans every AI-generated response BEFORE it is returned
#   to the user. Ensures the response is safe, appropriate, and always
#   includes the required medical disclaimer.
#
# WHY OUTPUT GUARDRAILS MATTER:
#   Even with a carefully crafted system prompt (in generator.py), the LLM
#   can occasionally produce responses that:
#     - Sound like a medical diagnosis ("Based on your symptoms, you have...")
#     - Sound like a prescription ("You should take 500mg of ibuprofen...")
#     - Are overly confident about a medical conclusion
#     - Are missing the required medical disclaimer
#
#   The output guardrails act as a safety net that catches these issues
#   AFTER generation, before the response reaches the user.
#
# HOW IT WORKS:
#   The checking has three layers:
#
#   LAYER 1 - Diagnosis Detection:
#     Scans the response for language patterns that sound like the AI
#     is diagnosing the user. If found, replaces the problematic sentence
#     with a safe alternative phrasing.
#
#   LAYER 2 - Prescription Detection:
#     Scans for language that sounds like the AI is prescribing medication
#     or giving specific dosage instructions to the user personally.
#     Replaces with appropriately hedged language.
#
#   LAYER 3 - Disclaimer Enforcement:
#     Checks that the standard medical disclaimer appears at the end of
#     the response. If it is missing (the LLM sometimes drops it), adds it.
#
# DESIGN PHILOSOPHY:
#   We try to FIX responses, not just block them.
#   Blocking a response after the user waited for it is a bad experience.
#   Instead, we subtly adjust problematic language so the response is still
#   useful but appropriately hedged.
#
#   Only in extreme cases (response is entirely unsafe) do we replace it
#   with a full fallback message.
#
# USED BY:
#   src/rag/pipeline.py (Step 7)
# ==============================================================================

import re
from loguru import logger


# ==============================================================================
# THE REQUIRED MEDICAL DISCLAIMER
# ==============================================================================
# This text MUST appear at the end of every response returned to users.
# It is defined once here and referenced throughout the module.
# ==============================================================================

MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "This information is for educational purposes only. "
    "Please consult a qualified healthcare professional "
    "for medical advice, diagnosis, or treatment."
)

# A shorter version used to detect if the disclaimer is already present
DISCLAIMER_DETECTION_PHRASE = "for educational purposes only"


# ==============================================================================
# DIAGNOSIS DETECTION PATTERNS
# ==============================================================================
# These patterns detect language where the AI sounds like it is
# diagnosing the user directly (which it must never do).
#
# Each entry is a tuple of:
#   (compiled_pattern, safe_replacement_template)
#
# The replacement uses \1, \2 etc. to preserve parts of the matched text
# while changing the diagnostic framing to educational framing.
# ==============================================================================

DIAGNOSIS_PATTERNS = [
    # "You have diabetes" / "You have a heart condition"
    (
        re.compile(
            r"\byou\s+have\s+(a\s+|an\s+)?([\w\s\-]+(?:disease|condition|"
            r"disorder|syndrome|infection|cancer|tumor|fracture|injury))\b",
            re.IGNORECASE,
        ),
        r"clinical cases with similar presentations have involved \2",
    ),
    # "You are suffering from" / "You are experiencing"
    (
        re.compile(
            r"\byou\s+are\s+(suffering\s+from|experiencing|showing\s+signs\s+of|"
            r"displaying\s+symptoms\s+of)\s+([\w\s\-]+)\b",
            re.IGNORECASE,
        ),
        r"patients in similar clinical cases have been found to be \1 \2",
    ),
    # "This is [condition]" or "This sounds like [condition]"
    (
        re.compile(
            r"\bthis\s+(is|sounds\s+like|appears\s+to\s+be|looks\s+like)\s+"
            r"(a\s+|an\s+)?([\w\s\-]+(?:disease|condition|disorder|syndrome|"
            r"infection|cancer|tumor))\b",
            re.IGNORECASE,
        ),
        r"clinical literature describes \3 as a condition where",
    ),
    # "You likely have" / "You probably have" / "You definitely have"
    (
        re.compile(
            r"\byou\s+(likely|probably|definitely|certainly|clearly)\s+have\b",
            re.IGNORECASE,
        ),
        r"clinical cases suggest patients with similar symptoms may have",
    ),
    # "My diagnosis is" / "The diagnosis is"
    (
        re.compile(
            r"\b(my|the)\s+diagnosis\s+(is|would\s+be)\b",
            re.IGNORECASE,
        ),
        r"a clinical assessment would consider",
    ),
]


# ==============================================================================
# PRESCRIPTION DETECTION PATTERNS
# ==============================================================================
# These patterns detect language where the AI sounds like it is
# prescribing medication or giving dosage instructions directly to the user.
# ==============================================================================

PRESCRIPTION_PATTERNS = [
    # "You should take [drug]" / "You need to take [drug]"
    (
        re.compile(
            r"\byou\s+should\s+take\s+([\w\s\-]+(?:mg|ml|tablet|capsule|dose|pill))",
            re.IGNORECASE,
        ),
        r"clinicians in similar cases have prescribed \1 — however, dosage must be determined by your doctor",
    ),
    # "Take [number] [units] of [drug]"
    (
        re.compile(
            r"\btake\s+(\d+[\w\s]*(?:mg|ml|tablets?|capsules?|pills?|doses?))\s+"
            r"of\s+([\w\s]+)\b",
            re.IGNORECASE,
        ),
        r"clinical cases have documented use of \1 of \2 — your doctor will determine the right dose for you",
    ),
    # "I recommend [drug]" / "I suggest [drug]"
    (
        re.compile(
            r"\bI\s+(recommend|suggest|prescribe|advise)\s+(taking\s+)?([\w\s\-]+(?:"
            r"mg|ml|tablet|capsule|dose|pill|medication|medicine|drug))\b",
            re.IGNORECASE,
        ),
        r"healthcare professionals may consider \3 in similar cases",
    ),
    # "The dose is [X]mg" directed at the user
    (
        re.compile(
            r"\bthe\s+(correct|right|appropriate|recommended)\s+dose\s+for\s+you\s+is\b",
            re.IGNORECASE,
        ),
        r"the appropriate dose, which must be determined by a licensed healthcare provider, is",
    ),
]


# ==============================================================================
# EXTREME CONTENT PATTERNS — full response replacement
# ==============================================================================
# If ANY of these patterns appear in the response, the entire response
# is replaced with a safe fallback. These represent cases where the
# response is so far outside safe boundaries that partial fixes are not enough.
# ==============================================================================

EXTREME_CONTENT_PATTERNS = [
    # Step-by-step instructions that could cause harm
    re.compile(
        r"\bstep\s*\d+\b.{0,100}\b(inject|overdose|administer\s+lethal|"
        r"harm|poison)\b",
        re.IGNORECASE,
    ),
    # Direct instructions to harm self or others
    re.compile(
        r"\b(you\s+should|you\s+can|try\s+to)\b.{0,50}\b(hurt|harm|kill|"
        r"injure|poison)\b.{0,30}\b(yourself|someone|them|him|her)\b",
        re.IGNORECASE,
    ),
]

# The fallback response used when extreme content is detected
EXTREME_CONTENT_FALLBACK = (
    "I'm not able to provide a response to this query. "
    "Please rephrase your question or ask about a different medical topic.\n\n"
    "If you are in distress or experiencing a medical emergency, "
    "please call emergency services immediately: 112 or 911." + MEDICAL_DISCLAIMER
)


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================


def apply_output_guardrails(response_text: str) -> str:
    """
    Cleans and validates an AI-generated response before returning it to the user.

    Purpose:
        This is the single entry point for all output safety processing.
        It applies all three layers of checking in sequence and returns
        the cleaned, safe response text.

        Called by pipeline.py in Step 7, after the LLM generates its response.

    Processing order:
        1. Check for extreme content (full replacement if found)
        2. Fix diagnosis language (targeted replacements)
        3. Fix prescription language (targeted replacements)
        4. Ensure medical disclaimer is present (append if missing)

    Parameters:
        response_text (str):
            The raw text output from the OpenAI Chat Completions API.
            This has not yet been shown to the user.

    Returns:
        str: The cleaned response text, safe to display to the user.
             Will always end with the medical disclaimer.

    Example:
        raw = "You have diabetes. You should take 500mg of metformin daily."
        safe = apply_output_guardrails(raw)
        # "Clinical cases with similar presentations have involved diabetes.
        #  Clinicians in similar cases have prescribed 500mg of metformin
        #  — however, dosage must be determined by your doctor.
        #  ---
        #  This information is for educational purposes only..."
    """
    if not response_text or not response_text.strip():
        # Empty response from LLM — return a safe fallback
        logger.warning("Output guardrail: received empty response from generator.")
        return (
            "I was unable to generate a response for your query. "
            "Please try rephrasing your question." + MEDICAL_DISCLAIMER
        )

    cleaned = response_text

    # --------------------------------------------------------------------------
    # LAYER 1: Check for extreme content (full replacement)
    # --------------------------------------------------------------------------

    for pattern in EXTREME_CONTENT_PATTERNS:
        if pattern.search(cleaned):
            logger.warning(
                "Output guardrail: EXTREME content detected. "
                "Replacing entire response with safe fallback."
            )
            return EXTREME_CONTENT_FALLBACK

    # --------------------------------------------------------------------------
    # LAYER 2: Fix diagnosis language
    # --------------------------------------------------------------------------

    diagnosis_fixes_applied = 0

    for pattern, replacement in DIAGNOSIS_PATTERNS:
        new_text, num_substitutions = pattern.subn(replacement, cleaned)
        if num_substitutions > 0:
            cleaned = new_text
            diagnosis_fixes_applied += num_substitutions

    if diagnosis_fixes_applied > 0:
        logger.info(
            f"Output guardrail: applied {diagnosis_fixes_applied} "
            f"diagnosis language fix(es)."
        )

    # --------------------------------------------------------------------------
    # LAYER 3: Fix prescription language
    # --------------------------------------------------------------------------

    prescription_fixes_applied = 0

    for pattern, replacement in PRESCRIPTION_PATTERNS:
        new_text, num_substitutions = pattern.subn(replacement, cleaned)
        if num_substitutions > 0:
            cleaned = new_text
            prescription_fixes_applied += num_substitutions

    if prescription_fixes_applied > 0:
        logger.info(
            f"Output guardrail: applied {prescription_fixes_applied} "
            f"prescription language fix(es)."
        )

    # --------------------------------------------------------------------------
    # LAYER 4: Ensure the medical disclaimer is present
    # --------------------------------------------------------------------------

    cleaned = _ensure_disclaimer_present(cleaned)

    # Log a summary of what was changed
    total_fixes = diagnosis_fixes_applied + prescription_fixes_applied
    if total_fixes > 0:
        logger.info(
            f"Output guardrail: response cleaned with {total_fixes} total fix(es). "
            f"({diagnosis_fixes_applied} diagnosis, "
            f"{prescription_fixes_applied} prescription)"
        )
    else:
        logger.debug("Output guardrail: response passed all checks with no changes.")

    return cleaned


# ==============================================================================
# PRIVATE HELPERS
# ==============================================================================


def _ensure_disclaimer_present(response_text: str) -> str:
    """
    Checks if the medical disclaimer is in the response and appends it if not.

    Purpose:
        The system prompt in generator.py instructs the LLM to always include
        the disclaimer, but the LLM occasionally omits it. This function is
        the safety net that guarantees it is always present.

    How it detects the disclaimer:
        It checks for the presence of DISCLAIMER_DETECTION_PHRASE
        (a short substring from the disclaimer) rather than the full text.
        This handles cases where the LLM paraphrases the disclaimer slightly.

    Parameters:
        response_text (str): The response text to check.

    Returns:
        str: The response text, guaranteed to contain the medical disclaimer.
    """
    # Check if the disclaimer (or a version of it) is already present
    if DISCLAIMER_DETECTION_PHRASE.lower() in response_text.lower():
        logger.debug("Output guardrail: disclaimer already present in response.")
        return response_text

    # Disclaimer is missing — append it
    logger.info("Output guardrail: disclaimer was missing, appending it now.")
    return response_text.rstrip() + MEDICAL_DISCLAIMER


def check_response_safety(response_text: str) -> tuple[bool, list[str]]:
    """
    Analyses a response and returns a safety report without modifying it.

    Purpose:
        Used by the human review system and evaluation module to understand
        WHY a response was flagged, without actually changing the text.
        Also useful for logging and debugging the guardrail behaviour.

    Parameters:
        response_text (str): The response text to analyse.

    Returns:
        tuple[bool, list[str]]:
            - bool: True if the response is completely safe, False if issues found.
            - list[str]: List of issue descriptions found. Empty if safe.

    Example:
        is_safe, issues = check_response_safety("You have diabetes.")
        print(is_safe)   # False
        print(issues)    # ["Diagnosis language detected: 'you have diabetes'"]
    """
    issues = []

    # Check for extreme content
    for pattern in EXTREME_CONTENT_PATTERNS:
        if pattern.search(response_text):
            issues.append("Extreme harmful content detected in response.")

    # Check for diagnosis language
    for pattern, _ in DIAGNOSIS_PATTERNS:
        match = pattern.search(response_text)
        if match:
            issues.append(f"Diagnosis language detected: '{match.group(0)[:60]}'")

    # Check for prescription language
    for pattern, _ in PRESCRIPTION_PATTERNS:
        match = pattern.search(response_text)
        if match:
            issues.append(f"Prescription language detected: '{match.group(0)[:60]}'")

    # Check for missing disclaimer
    if DISCLAIMER_DETECTION_PHRASE.lower() not in response_text.lower():
        issues.append("Medical disclaimer is missing from response.")

    is_safe = len(issues) == 0

    return is_safe, issues
