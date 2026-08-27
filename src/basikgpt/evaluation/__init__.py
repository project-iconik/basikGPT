"""Language model evaluation subsystem for basikGPT."""

from basikgpt.evaluation.hellaswag import (
    CandidateScore,
    HellaSwagExample,
    HellaSwagResult,
    HellaSwagSummary,
    evaluate_hellaswag,
    evaluate_hellaswag_example,
    format_hellaswag_context,
    load_hellaswag_dataset,
    score_completion,
)
from basikgpt.evaluation.language_model import (
    evaluate_language_model,
    save_evaluation_result,
)

__all__ = [
    "evaluate_language_model",
    "save_evaluation_result",
    "CandidateScore",
    "HellaSwagExample",
    "HellaSwagResult",
    "HellaSwagSummary",
    "evaluate_hellaswag",
    "evaluate_hellaswag_example",
    "format_hellaswag_context",
    "score_completion",
    "load_hellaswag_dataset",
]
