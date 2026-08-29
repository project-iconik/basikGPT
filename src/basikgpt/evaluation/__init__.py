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
from basikgpt.evaluation.lambada import (
    LambadaResult,
    LambadaSummary,
    evaluate_lambada,
    evaluate_lambada_example,
    split_last_word,
)
from basikgpt.evaluation.language_model import (
    evaluate_language_model,
    save_evaluation_result,
)
from basikgpt.evaluation.multiple_choice import (
    MultipleChoiceResult,
    MultipleChoiceSummary,
    evaluate_multiple_choice,
    extract_logits,
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
    "LambadaResult",
    "LambadaSummary",
    "evaluate_lambada",
    "evaluate_lambada_example",
    "split_last_word",
    "MultipleChoiceResult",
    "MultipleChoiceSummary",
    "evaluate_multiple_choice",
    "extract_logits",
]
