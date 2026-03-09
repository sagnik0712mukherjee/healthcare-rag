# ==============================================================================
# src/rag/generator.py
# ==============================================================================
# PURPOSE:
#   Takes a user query + retrieved context chunks and generates a safe,
#   contextual medical response using the OpenAI Chat API (GPT-4o-mini).
#
# WHAT THE GENERATOR DOES:
#   1. Formats the retrieved chunks into a readable context block
#   2. Builds a carefully structured system + user prompt
#   3. Injects the conversation history (short-term memory) for context
#   4. Calls the OpenAI Chat Completions API
#   5. Returns the response text AND token usage counts
#
# WHY TOKEN COUNTS MATTER:
#   We return input_tokens and output_tokens alongside the response text
#   so the monitoring module can log them and deduct from the user's budget.
#
# THE SYSTEM PROMPT IS CRITICAL:
#   The system prompt instructs the LLM to:
#   - Only answer based on the retrieved context (reduce hallucination)
#   - Never diagnose or prescribe (safety)
#   - Always acknowledge uncertainty
#   - Refer users to healthcare professionals
#
# USED BY:
#   src/rag/pipeline.py
# ==============================================================================

from openai import OpenAI
from loguru import logger
from dataclasses import dataclass

from config.settings import settings


# Create a single shared OpenAI client for this module
openai_client = OpenAI(api_key=settings.openai_api_key)

# ------------------------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------------------------
# This is the most important piece of text in the entire system.
# It tells the LLM exactly how to behave as a healthcare assistant.
# It is injected as the "system" role in every API call.
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful, knowledgeable, and cautious healthcare information assistant.

Your purpose is to help patients and healthcare users understand medical topics by drawing on clinical case information and medical literature.

RULES YOU MUST ALWAYS FOLLOW:

1. ONLY use the information provided in the CONTEXT section below to answer questions.
   If the context does not contain enough information to answer, say so honestly.
   Do not invent or guess medical facts.

2. NEVER diagnose a patient. Do not say "You have [condition]" or "This sounds like [disease]".
   Instead, describe what clinical cases show or what symptoms are generally associated with conditions.

3. NEVER prescribe or recommend specific medications or dosages.
   You may describe what medications are mentioned in clinical cases, but always frame this
   as "clinicians in similar cases have used..." rather than "you should take...".

4. ALWAYS acknowledge uncertainty. Use phrases like:
   - "Based on the clinical cases in my knowledge base..."
   - "Clinical literature suggests..."
   - "This is for educational purposes only..."

5. If the user describes a personal emergency (e.g., chest pain right now, difficulty breathing),
   immediately tell them to call emergency services (112 / 911) before anything else.

6. Be clear, compassionate, and use plain language that patients can understand.
   Avoid excessive medical jargon unless the user appears to be a healthcare professional.

7. ALWAYS end your response with the standard medical disclaimer.

STANDARD DISCLAIMER (include this at the end of EVERY response):
---
This information is for educational purposes only. Please consult a qualified healthcare professional for medical advice, diagnosis, or treatment.
---"""


# ------------------------------------------------------------------------------
# RESPONSE DATACLASS
# ------------------------------------------------------------------------------


@dataclass
class GeneratorResponse:
    """
    Container for the output of a single LLM generation call.

    Purpose:
        Bundles the response text and token usage counts into a single
        object that pipeline.py can use cleanly.

    Attributes:
        response_text (str): The AI-generated response to return to the user.
        input_tokens (int): Number of tokens in the prompt (system + context + query).
        output_tokens (int): Number of tokens in the generated response.
        total_tokens (int): input_tokens + output_tokens.
        model_used (str): The OpenAI model name that generated this response.
    """

    response_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_used: str


# ------------------------------------------------------------------------------
# MAIN GENERATION FUNCTION
# ------------------------------------------------------------------------------


def generate_response(
    query: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict] = None,
) -> GeneratorResponse:
    """
    Generates a medical response using retrieved context and the OpenAI API.

    Purpose:
        This is the "G" in RAG (Generation). It takes the user's query and
        the context chunks retrieved by the retriever, formats them into
        a prompt, and calls the OpenAI Chat Completions API to generate
        a response.

    Parameters:
        query (str):
            The user's current question.
            Example: "What are the early symptoms of diabetes?"
        retrieved_chunks (list[dict]):
            List of chunk dicts returned by retriever.retrieve().
            Each dict has at least "chunk_text" and "source" keys.
            These are formatted into the context block of the prompt.
        conversation_history (list[dict], optional):
            Previous turns in the conversation for this session.
            Each dict has "role" ("user" or "assistant") and "content" (str).
            This implements short-term conversational memory.
            Example:
            [
                {"role": "user", "content": "I have chest pain"},
                {"role": "assistant", "content": "How long have you had it?"}
            ]

    Returns:
        GeneratorResponse: A dataclass with:
            - response_text: The AI's answer
            - input_tokens: Tokens used in the prompt
            - output_tokens: Tokens used in the response
            - total_tokens: Total tokens consumed
            - model_used: The model name (e.g., "gpt-4o-mini")

    Example:
        chunks = retrieve("diabetes symptoms", top_k=5)
        result = generate_response(
            query="What are early warning signs of diabetes?",
            retrieved_chunks=chunks,
        )
        print(result.response_text)
        print(f"Tokens used: {result.total_tokens}")
    """
    if conversation_history is None:
        conversation_history = []

    # Step 1: Format the retrieved chunks into a context block
    context_block = _format_context(retrieved_chunks)

    # Step 2: Build the full user message including the context
    user_message_with_context = _build_user_message(
        query=query,
        context_block=context_block,
    )

    # Step 3: Assemble the messages list for the OpenAI API
    # Structure:
    #   [system prompt]
    #   [turn 1: user message]  <- from conversation_history
    #   [turn 1: assistant reply]
    #   ...
    #   [current user message with context]  <- new query
    messages = _build_messages(
        user_message_with_context=user_message_with_context,
        conversation_history=conversation_history,
    )

    # Step 4: Call the OpenAI Chat Completions API
    logger.info(
        f"Calling OpenAI API: model={settings.openai_chat_model}, "
        f"messages={len(messages)}, "
        f"query='{query[:60]}...'"
    )

    api_response = openai_client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        # Controls randomness: 0.0 = deterministic, 1.0 = creative
        # 0.3 gives consistent, factual medical answers with slight variation
        temperature=0.3,
        # Maximum tokens in the generated response
        # 1024 is enough for detailed medical explanations
        max_tokens=1024,
        # top_p=1.0 means no nucleus sampling cutoff (use full distribution)
        top_p=1.0,
    )

    # Step 5: Extract the response text and token usage from the API response
    response_text = api_response.choices[0].message.content

    input_tokens = api_response.usage.prompt_tokens
    output_tokens = api_response.usage.completion_tokens
    total_tokens = api_response.usage.total_tokens

    logger.info(
        f"Generation complete. "
        f"Input tokens: {input_tokens}, "
        f"Output tokens: {output_tokens}, "
        f"Total: {total_tokens}"
    )

    return GeneratorResponse(
        response_text=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model_used=settings.openai_chat_model,
    )


# ------------------------------------------------------------------------------
# PRIVATE HELPER FUNCTIONS
# ------------------------------------------------------------------------------


def _format_context(retrieved_chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a readable context block for the prompt.

    Purpose:
        Converts the list of chunk dicts into a structured text block
        that the LLM can read and reference in its response.
        Each chunk is numbered and labeled by source type.

    Format example:
        --- CONTEXT ---

        [Source 1 - Clinical Case | Patient: 45-year-old Male]
        A 45-year-old male presented with progressive shortness of breath...

        [Source 2 - Medical Image Caption | Type: radiology/x_ray]
        Chest X-ray showing bilateral pulmonary infiltrates consistent with...

        --- END CONTEXT ---

    Parameters:
        retrieved_chunks (list[dict]):
            List of chunk dicts from the retriever.

    Returns:
        str: A formatted multi-line string ready for prompt injection.
             Returns a "no context" message if the list is empty.
    """
    if not retrieved_chunks:
        return (
            "--- CONTEXT ---\n"
            "No relevant clinical cases or images were found for this query.\n"
            "Please answer based on general medical knowledge if possible,\n"
            "and clearly state that no specific case context was available.\n"
            "--- END CONTEXT ---"
        )

    context_parts = ["--- CONTEXT ---\n"]

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "unknown")
        score = chunk.get("similarity_score", 0.0)
        text = chunk.get("chunk_text", "").strip()

        # Build a source label that gives the LLM useful metadata
        if source == "clinical_case":
            age = chunk.get("patient_age")
            gender = chunk.get("patient_gender", "Unknown")

            # Format age as "45-year-old" or "Unknown age" if missing
            age_str = f"{age}-year-old" if age else "Unknown age"

            source_label = (
                f"[Source {i} - Clinical Case | "
                f"Patient: {age_str} {gender} | "
                f"Relevance: {score:.2f}]"
            )

        elif source == "image_caption":
            image_type = chunk.get("image_type", "unknown")
            image_subtype = chunk.get("image_subtype", "")
            labels = chunk.get("labels", [])

            # Build a compact label string from the image classification labels
            label_str = ", ".join(labels[:4]) if labels else "unclassified"

            type_str = f"{image_type}"
            if image_subtype and image_subtype != "unknown":
                type_str += f"/{image_subtype}"

            source_label = (
                f"[Source {i} - Medical Image Caption | "
                f"Type: {type_str} | "
                f"Labels: {label_str} | "
                f"Relevance: {score:.2f}]"
            )

        else:
            source_label = f"[Source {i} - {source} | Relevance: {score:.2f}]"

        context_parts.append(source_label)
        context_parts.append(text)
        context_parts.append("")  # Blank line between chunks

    context_parts.append("--- END CONTEXT ---")

    return "\n".join(context_parts)


def _build_user_message(query: str, context_block: str) -> str:
    """
    Combines the context block and user query into a single user message.

    Purpose:
        The OpenAI API works best when context is provided BEFORE the
        question, so the model can reference the context while forming
        its answer. This function produces a well-structured combined message.

    Parameters:
        query (str): The user's question.
        context_block (str): The formatted context from _format_context().

    Returns:
        str: A complete user message string ready to send to the API.
    """
    return (
        f"{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"Please answer the question using the context provided above. "
        f"If the context does not contain enough information, say so clearly."
    )


def _build_messages(
    user_message_with_context: str,
    conversation_history: list[dict],
) -> list[dict]:
    """
    Assembles the complete messages list for the OpenAI Chat Completions API.

    Purpose:
        The OpenAI Chat API takes a list of message dicts, each with a
        "role" ("system", "user", or "assistant") and "content" (string).
        This function builds that list in the correct order:
          1. System prompt (always first)
          2. Previous conversation turns (short-term memory)
          3. Current user message with context injected

    Why inject context only in the current message (not history)?
        Previous turns already have their own context from when they were
        generated. Re-injecting all context into history would waste tokens.
        We only inject context for the CURRENT query.

    Parameters:
        user_message_with_context (str): Current query + context block.
        conversation_history (list[dict]): Previous turns from short-term memory.

    Returns:
        list[dict]: Ordered list of message dicts for the OpenAI API.
    """
    messages = []

    # 1. System prompt is always the first message
    messages.append(
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    )

    # 2. Add previous conversation turns (short-term memory)
    # These give the LLM awareness of what was discussed earlier
    # so it can handle follow-up questions like "tell me more about that"
    for turn in conversation_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")

        # Only include valid roles and non-empty content
        if role in ("user", "assistant") and content.strip():
            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    # 3. Current user message (with context block injected)
    messages.append(
        {
            "role": "user",
            "content": user_message_with_context,
        }
    )

    return messages
