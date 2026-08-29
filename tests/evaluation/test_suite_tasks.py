"""Offline tests for PIQA / WinoGrande / ARC-Easy parsers and REPORT.md generation."""

from pathlib import Path

from basikgpt.evaluation.suite import PROTOCOL_MODELS, write_report
from basikgpt.evaluation.tasks import (
    parse_arc_example,
    parse_piqa_example,
    parse_winogrande_example,
)


def test_parse_piqa_example() -> None:
    doc = {
        "goal": "To make a pillow fluffier",
        "sol1": "put it in the dryer",
        "sol2": "put it in the freezer",
        "label": 0,
    }
    example_id, context, choices, gold = parse_piqa_example(doc)
    assert context == "To make a pillow fluffier"
    assert choices == ["put it in the dryer", "put it in the freezer"]
    assert gold == 0


def test_parse_winogrande_blank_option_plus_right() -> None:
    doc = {
        "qID": "ex1",
        "sentence": "The trophy doesn't fit in the suitcase because _ is too large.",
        "option1": "the trophy",
        "option2": "the suitcase",
        "answer": "1",
    }
    example_id, context, choices, gold = parse_winogrande_example(doc)
    assert example_id == "ex1"
    assert context == "The trophy doesn't fit in the suitcase because"
    assert choices[0] == "the trophy is too large."
    assert choices[1] == "the suitcase is too large."
    assert gold == 0


def test_parse_arc_easy_answer_key() -> None:
    doc = {
        "id": "ArcEasy-1",
        "question": "Which object is a conductor?",
        "choices": {
            "text": ["wood", "copper", "plastic", "glass"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
    }
    example_id, context, choices, gold = parse_arc_example(doc)
    assert example_id == "ArcEasy-1"
    assert context.startswith("Question: Which object is a conductor?")
    assert context.endswith("Answer:")
    assert choices[gold] == "copper"
    assert gold == 1


def test_write_report_contains_protocol_and_scores(tmp_path: Path) -> None:
    summary = {
        "protocol": "english-lm-suite-v1",
        "models": {
            "basikgpt-2p5b": {
                "id": "basikgpt-2p5b",
                "params_label": "124M",
                "family": "GPT-2 Small (basikGPT)",
                "corpus": "FineWeb-Edu 2.5B tokens",
                "tasks": {
                    "hellaswag": {"acc_norm": 0.2933, "acc_raw": 0.28},
                    "lambada_openai": {"accuracy": 0.40},
                    "piqa": {"acc_norm": 0.60},
                    "winogrande": {"acc_raw": 0.51},
                    "arc_easy": {"acc_norm": 0.45},
                },
            }
        },
    }
    path = write_report(tmp_path, summary)
    text = path.read_text(encoding="utf-8")
    assert path.name == "REPORT.md"
    assert "HellaSwag" in text
    assert "LAMBADA" in text
    assert "openai-community/gpt2" in text
    assert "Qwen/Qwen2.5-0.5B" in text
    assert "apple/OpenELM" not in text
    assert "OpenELM" in text  # listed under Not included
    assert "basikgpt-2p5b" in text
    assert "basikgpt-5b" in text
    assert "29.33%" in text
    assert "| Avg |" in text
    assert "45.07%" in text
    assert "unweighted mean of the five primaries" in text
    assert "runs/main_2p5b/step-00038147.pt" in text
    assert "runs/cont_5b_mix/step-00076294.pt" in text
    assert "acc_norm" in text
    assert "GPT-2 Medium" in text or "Not included" in text


def test_protocol_models_have_hub_ids() -> None:
    by_id = {spec.id: spec for spec in PROTOCOL_MODELS}
    assert by_id["basikgpt-2p5b"].hf_id == "project-iconik/basikGPT-1-v1.0"
    assert by_id["basikgpt-5b"].hf_id == "project-iconik/basikGPT-1-v1.1"
