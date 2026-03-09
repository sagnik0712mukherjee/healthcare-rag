# ==============================================================================
# src/evaluation/ragas_eval.py
# ==============================================================================
# PURPOSE:
#   Standalone evaluation script that measures the quality of the RAG system
#   using the RAGAS framework. Run this periodically to benchmark the system
#   and track quality changes after index updates or prompt modifications.
#
# WHAT IS RAGAS?
#   RAGAS (Retrieval-Augmented Generation Assessment) is an evaluation
#   framework specifically designed for RAG systems. It measures quality
#   across several dimensions without requiring human-labelled "gold standard"
#   answers — it uses an LLM to judge the results.
#
# THE THREE CORE METRICS:
#
#   1. FAITHFULNESS (0.0 - 1.0)
#      "Does the answer only contain claims that can be verified from the
#      retrieved context?"
#      High score = the AI isn't hallucinating.
#      Low score  = the AI is making things up not in the retrieved chunks.
#
#   2. ANSWER RELEVANCY (0.0 - 1.0)
#      "How well does the answer actually address the question asked?"
#      High score = the answer is directly relevant to the question.
#      Low score  = the answer is off-topic or incomplete.
#
#   3. CONTEXT RECALL (0.0 - 1.0)
#      "Does the retrieved context contain the information needed to
#      answer the question?"
#      High score = the retrieval is finding the right chunks.
#      Low score  = the FAISS search is missing relevant content.
#      Note: Requires a ground_truth (reference answer) to measure.
#
# HOW TO RUN:
#   # Evaluate with built-in sample questions:
#   python src/evaluation/ragas_eval.py
#
#   # Evaluate with a custom questions CSV file:
#   python src/evaluation/ragas_eval.py --questions data/eval_questions.csv
#
#   # Specify number of samples:
#   python src/evaluation/ragas_eval.py --samples 20
#
#   # Save results to a JSON file:
#   python src/evaluation/ragas_eval.py --output results/eval_2025_03_09.json
#
# CSV FORMAT (for --questions):
#   question,ground_truth
#   "What are symptoms of diabetes?","Diabetes symptoms include..."
#   "What causes hypertension?","Hypertension is caused by..."
#   The ground_truth column is optional but required for context_recall.
#
# PREREQUISITES:
#   - FAISS index must be built (run build_faiss_index.py first)
#   - OPENAI_API_KEY must be set in .env
#   - DATABASE_URL must be set in .env
#
# USED BY:
#   Run standalone — not imported by the main application.
# ==============================================================================

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.rag.retriever import retrieve
from src.rag.generator import generate_response

from loguru import logger

# Add project root to sys.path so imports work when run directly
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# These imports require the package to be installed (see requirements.txt)
# ragas==0.1.14, datasets (from HuggingFace)
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_recall, faithfulness

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning(
        "RAGAS or datasets library not installed. "
        "Install with: pip install ragas datasets"
    )


# ==============================================================================
# BUILT-IN SAMPLE EVALUATION QUESTIONS
# ==============================================================================
# These are 20 representative medical questions covering a range of topics
# likely to appear in the MultiCaRe dataset. Used when no custom question
# file is provided.
#
# Format: list of dicts with "question" and optional "ground_truth"
# ==============================================================================

SAMPLE_EVAL_QUESTIONS = [
    {
        "question": "What are the early symptoms of type 2 diabetes?",
        "ground_truth": (
            "Early symptoms of type 2 diabetes include increased thirst, "
            "frequent urination, unexplained weight loss, fatigue, blurred vision, "
            "slow-healing sores, and frequent infections."
        ),
    },
    {
        "question": "What causes hypertension and how is it managed?",
        "ground_truth": (
            "Hypertension is caused by a combination of genetic factors, lifestyle "
            "factors such as high salt intake, obesity, physical inactivity, "
            "and stress. Management includes lifestyle changes and antihypertensive medications."
        ),
    },
    {
        "question": "What are the clinical features of pulmonary tuberculosis?",
        "ground_truth": (
            "Pulmonary tuberculosis presents with persistent cough (often with blood), "
            "fever, night sweats, weight loss, and fatigue. Chest X-ray may show "
            "upper lobe infiltrates or cavitations."
        ),
    },
    {
        "question": "What are the signs and symptoms of myocardial infarction?",
        "ground_truth": (
            "Myocardial infarction typically presents with chest pain or pressure "
            "radiating to the arm or jaw, shortness of breath, sweating, nausea, "
            "and may include ECG changes and elevated cardiac enzymes."
        ),
    },
    {
        "question": "How is pneumonia diagnosed in clinical practice?",
        "ground_truth": (
            "Pneumonia is diagnosed based on clinical presentation (fever, cough, "
            "dyspnoea), physical examination findings (decreased breath sounds, "
            "crackles), chest X-ray showing consolidation, and laboratory markers "
            "such as elevated white blood cell count."
        ),
    },
    {
        "question": "What are the common causes of acute abdominal pain?",
        "ground_truth": (
            "Common causes of acute abdominal pain include appendicitis, "
            "cholecystitis, peptic ulcer disease, pancreatitis, bowel obstruction, "
            "renal colic, and ectopic pregnancy."
        ),
    },
    {
        "question": "What imaging findings are associated with a pulmonary embolism?",
        "ground_truth": (
            "Pulmonary embolism may show normal chest X-ray or subtle findings like "
            "Hampton's hump or Westermark sign. CT pulmonary angiography is the "
            "gold standard showing filling defects in pulmonary arteries."
        ),
    },
    {
        "question": "What are the neurological signs of a stroke?",
        "ground_truth": (
            "Stroke presents with sudden onset of facial drooping, arm weakness, "
            "speech difficulties (FAST criteria), along with possible visual changes, "
            "severe headache, dizziness, and loss of coordination."
        ),
    },
    {
        "question": "How is sepsis defined and what are its management principles?",
        "ground_truth": (
            "Sepsis is life-threatening organ dysfunction caused by a dysregulated "
            "host response to infection. Management includes early fluid resuscitation, "
            "blood cultures, broad-spectrum antibiotics within one hour, and source control."
        ),
    },
    {
        "question": "What are the radiological features of lung cancer?",
        "ground_truth": (
            "Lung cancer may appear on chest X-ray or CT as a solitary pulmonary nodule, "
            "mass, hilar enlargement, pleural effusion, or consolidation. "
            "CT is used for staging and PET scan for metastasis assessment."
        ),
    },
    {
        "question": "What are the classic presentation features of appendicitis?",
        "ground_truth": (
            "Appendicitis classically presents with periumbilical pain migrating to "
            "the right iliac fossa, nausea, vomiting, fever, and elevated white "
            "blood cell count. Rovsing's sign and rebound tenderness may be present."
        ),
    },
    {
        "question": "What are the clinical features of Crohn's disease?",
        "ground_truth": (
            "Crohn's disease presents with abdominal pain, diarrhoea (sometimes bloody), "
            "weight loss, fatigue, and perianal disease. It can affect any part of the "
            "gastrointestinal tract with transmural inflammation."
        ),
    },
    {
        "question": "What is the typical presentation of rheumatoid arthritis?",
        "ground_truth": (
            "Rheumatoid arthritis presents with symmetric joint inflammation typically "
            "affecting the small joints of the hands and feet, morning stiffness lasting "
            "more than an hour, joint swelling, and elevated inflammatory markers."
        ),
    },
    {
        "question": "How do clinicians distinguish between Type 1 and Type 2 diabetes?",
        "ground_truth": (
            "Type 1 diabetes typically presents in younger patients with rapid onset, "
            "weight loss, and ketoacidosis. Type 2 is more common in older, overweight "
            "patients with gradual onset. C-peptide levels and autoantibodies help differentiate."
        ),
    },
    {
        "question": "What are the MRI findings in multiple sclerosis?",
        "ground_truth": (
            "MRI in multiple sclerosis typically shows periventricular and juxtacortical "
            "white matter lesions, often described as Dawson's fingers on sagittal views. "
            "Gadolinium-enhancing lesions indicate active inflammation."
        ),
    },
    {
        "question": "What are the risk factors for deep vein thrombosis?",
        "ground_truth": (
            "DVT risk factors include prolonged immobility, recent surgery, malignancy, "
            "pregnancy, hormonal contraception, obesity, previous DVT, thrombophilia, "
            "and dehydration (Virchow's triad: stasis, hypercoagulability, endothelial damage)."
        ),
    },
    {
        "question": "How is acute kidney injury classified and managed?",
        "ground_truth": (
            "AKI is classified using KDIGO criteria based on creatinine rise and urine "
            "output. Management includes identifying and treating the cause, fluid "
            "resuscitation for prerenal causes, avoiding nephrotoxins, and dialysis "
            "if severe."
        ),
    },
    {
        "question": "What are the clinical manifestations of systemic lupus erythematosus?",
        "ground_truth": (
            "SLE manifests with butterfly rash, photosensitivity, oral ulcers, arthritis, "
            "serositis, renal disease, haematological abnormalities, and neurological "
            "features. ANA and anti-dsDNA antibodies support the diagnosis."
        ),
    },
    {
        "question": "What causes liver cirrhosis and what are its complications?",
        "ground_truth": (
            "Liver cirrhosis is caused by alcohol, chronic hepatitis B/C, NASH, "
            "and autoimmune hepatitis. Complications include portal hypertension, "
            "ascites, varices, hepatic encephalopathy, and hepatocellular carcinoma."
        ),
    },
    {
        "question": "What are the echocardiographic findings in heart failure with reduced ejection fraction?",
        "ground_truth": (
            "HFrEF (EF < 40%) shows dilated left ventricle, reduced ejection fraction, "
            "wall motion abnormalities, diastolic dysfunction, and possibly mitral "
            "regurgitation and elevated filling pressures on echocardiography."
        ),
    },
]


# ==============================================================================
# CORE EVALUATION FUNCTIONS
# ==============================================================================


def run_rag_for_question(question: str) -> dict:
    """
    Runs the full RAG pipeline (retrieve + generate) for a single evaluation question.

    Purpose:
        Used by the evaluator to get both the generated answer and the
        retrieved context chunks for a question. RAGAS needs both the
        answer AND the source contexts to compute its metrics.

    Parameters:
        question (str): The evaluation question.

    Returns:
        dict with:
            - question (str)
            - answer (str): The generated AI response
            - contexts (list[str]): The retrieved chunk texts
            - error (str | None): Error message if something failed
    """
    try:
        # Step 1: Retrieve relevant chunks
        chunks = retrieve(query=question, top_k=5)
        contexts = [chunk["chunk_text"] for chunk in chunks]

        if not contexts:
            logger.warning(f"No chunks retrieved for question: '{question[:60]}'")
            return {
                "question": question,
                "answer": "No relevant information found in the knowledge base.",
                "contexts": [],
                "error": "no_chunks_retrieved",
            }

        # Step 2: Generate response (no conversation history for evaluation)
        gen_result = generate_response(
            query=question,
            retrieved_chunks=chunks,
            conversation_history=[],
        )

        logger.info(
            f"Eval question processed: '{question[:60]}...' | "
            f"tokens={gen_result.total_tokens} | "
            f"chunks={len(chunks)}"
        )

        return {
            "question": question,
            "answer": gen_result.response_text,
            "contexts": contexts,
            "error": None,
        }

    except Exception as error:
        logger.error(f"Failed to process eval question '{question[:60]}': {error}")
        return {
            "question": question,
            "answer": "",
            "contexts": [],
            "error": str(error),
        }


def build_ragas_dataset(
    eval_questions: list[dict],
) -> Optional["Dataset"]:
    """
    Builds a HuggingFace Dataset object in the format RAGAS expects.

    Purpose:
        RAGAS's evaluate() function requires a HuggingFace Dataset with
        specific column names. This function runs the RAG pipeline on each
        evaluation question and assembles the results into that format.

    RAGAS expects these columns:
        - question: The input question
        - answer: The generated answer
        - contexts: List of retrieved context strings
        - ground_truth: Reference answer (optional, needed for context_recall)

    Parameters:
        eval_questions (list[dict]):
            List of {"question": str, "ground_truth": str (optional)} dicts.

    Returns:
        Dataset: HuggingFace Dataset ready for RAGAS evaluate().
        None: If dataset construction fails entirely.
    """
    if not RAGAS_AVAILABLE:
        logger.error("RAGAS is not available. Cannot build evaluation dataset.")
        return None

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    total = len(eval_questions)
    logger.info(f"Running RAG pipeline on {total} evaluation questions...")

    failed_count = 0

    for i, item in enumerate(eval_questions, start=1):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")

        if not question.strip():
            logger.warning(f"Skipping empty question at index {i}.")
            continue

        logger.info(f"Processing question {i}/{total}: '{question[:60]}...'")

        result = run_rag_for_question(question)

        if result["error"] and not result["answer"]:
            # Skip questions that completely failed (no answer, no contexts)
            failed_count += 1
            continue

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])

        # Use provided ground_truth or empty string (context_recall metric
        # will be skipped for questions without ground truths)
        ground_truths.append(ground_truth if ground_truth else "")

    if not questions:
        logger.error("All evaluation questions failed. Cannot build dataset.")
        return None

    logger.info(f"Dataset built: {len(questions)} successful, {failed_count} failed.")

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation(
    eval_questions: Optional[list[dict]] = None,
    output_path: Optional[str] = None,
) -> Optional[dict]:
    """
    Runs the complete RAGAS evaluation and returns/saves the scores.

    Purpose:
        The main entry point for evaluation. Builds the dataset, runs RAGAS
        metrics, logs results, and optionally saves to a JSON file.

    Parameters:
        eval_questions (list[dict], optional):
            List of {"question": str, "ground_truth": str} dicts.
            If None, uses SAMPLE_EVAL_QUESTIONS.
        output_path (str, optional):
            File path to save JSON results. If None, results are only logged.

    Returns:
        dict: RAGAS scores with keys "faithfulness", "answer_relevancy",
              "context_recall", plus metadata.
        None: If evaluation fails.
    """
    if not RAGAS_AVAILABLE:
        logger.error(
            "Cannot run evaluation: RAGAS is not installed.\n"
            "Install with: pip install ragas datasets"
        )
        return None

    if eval_questions is None:
        eval_questions = SAMPLE_EVAL_QUESTIONS
        logger.info(f"Using {len(eval_questions)} built-in sample questions.")
    else:
        logger.info(f"Using {len(eval_questions)} custom evaluation questions.")

    # Build the HuggingFace dataset by running the RAG pipeline on all questions
    dataset = build_ragas_dataset(eval_questions)

    if dataset is None:
        logger.error("Failed to build evaluation dataset. Aborting.")
        return None

    logger.info(f"Running RAGAS evaluation on {len(dataset)} samples...")

    # Determine which metrics to run
    # Only run context_recall if ground truths are present
    has_ground_truths = any(
        item.get("ground_truth", "").strip() for item in eval_questions
    )

    metrics = [faithfulness, answer_relevancy]
    if has_ground_truths:
        metrics.append(context_recall)
        logger.info("Ground truths detected — including context_recall metric.")
    else:
        logger.info(
            "No ground truths provided — skipping context_recall metric. "
            "Add 'ground_truth' to eval questions for a complete evaluation."
        )

    try:
        # Run RAGAS evaluation
        # This makes LLM calls to an OpenAI model to judge the results
        ragas_results = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        logger.info("=" * 60)
        logger.info("RAGAS EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Faithfulness:      {ragas_results['faithfulness']:.4f}")
        logger.info(f"Answer Relevancy:  {ragas_results['answer_relevancy']:.4f}")
        if has_ground_truths:
            logger.info(f"Context Recall:    {ragas_results['context_recall']:.4f}")
        logger.info("=" * 60)
        logger.info("Score interpretation: 0.0 = worst, 1.0 = best")
        logger.info("Target: Faithfulness > 0.85, Relevancy > 0.80, Recall > 0.75")
        logger.info("=" * 60)

        # Build results dict with metadata
        results = {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "num_questions": len(dataset),
            "model_used": settings.openai_chat_model,
            "embedding_model": settings.openai_embedding_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "scores": {
                "faithfulness": round(float(ragas_results["faithfulness"]), 4),
                "answer_relevancy": round(float(ragas_results["answer_relevancy"]), 4),
            },
            "metrics_run": [m.name for m in metrics],
        }

        if has_ground_truths:
            results["scores"]["context_recall"] = round(
                float(ragas_results["context_recall"]), 4
            )

        # Save to JSON if output path provided
        if output_path:
            _save_results_to_json(results, output_path)

        return results

    except Exception as error:
        logger.error(f"RAGAS evaluation failed: {error}")
        return None


def load_questions_from_csv(csv_path: str) -> list[dict]:
    """
    Loads evaluation questions from a CSV file.

    Expected CSV format:
        question,ground_truth
        "What are symptoms of diabetes?","Symptoms include..."
        "What causes hypertension?",""

    The ground_truth column is optional.
    Rows with empty questions are skipped.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        list[dict]: List of {"question": str, "ground_truth": str} dicts.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV has no 'question' column.
    """
    import csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    questions = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "question" not in reader.fieldnames:
            raise ValueError(
                f"CSV file must have a 'question' column. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            question = row.get("question", "").strip()
            ground_truth = row.get("ground_truth", "").strip()

            if question:
                questions.append(
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                    }
                )

    logger.info(f"Loaded {len(questions)} questions from {csv_path}")
    return questions


# ==============================================================================
# PRIVATE HELPERS
# ==============================================================================


def _save_results_to_json(results: dict, output_path: str) -> None:
    """
    Saves evaluation results to a JSON file.

    Creates parent directories if they don't exist.

    Parameters:
        results (dict): The evaluation results to save.
        output_path (str): The file path to save to.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to: {output_path}")

    except Exception as error:
        logger.error(f"Failed to save evaluation results: {error}")


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================


def _parse_args():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the Healthcare RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with built-in sample questions:
  python src/evaluation/ragas_eval.py

  # Run with a custom CSV file:
  python src/evaluation/ragas_eval.py --questions data/eval_questions.csv

  # Run with first 10 sample questions and save results:
  python src/evaluation/ragas_eval.py --samples 10 --output results/eval.json
        """,
    )

    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to a CSV file with evaluation questions (columns: question, ground_truth)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of built-in sample questions to use (default: all 20)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON (e.g., results/eval_2025_03_09.json)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Determine which questions to evaluate
    if args.questions:
        eval_questions = load_questions_from_csv(args.questions)
    else:
        eval_questions = SAMPLE_EVAL_QUESTIONS
        if args.samples:
            eval_questions = eval_questions[: args.samples]
            logger.info(f"Using first {args.samples} sample questions.")

    # Run the evaluation
    results = run_evaluation(
        eval_questions=eval_questions,
        output_path=args.output,
    )

    if results is None:
        logger.error("Evaluation failed. Check the logs above for details.")
        sys.exit(1)
    else:
        logger.info("Evaluation complete!")
        sys.exit(0)
