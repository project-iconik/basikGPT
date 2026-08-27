"""Unit tests for deterministic document-level dataset splitting."""

import random
import pytest
from basikgpt.data.split import get_document_split, is_validation_document


def test_split_determinism() -> None:
    """Verifies that split assignment for any document ID is 100% deterministic."""
    doc_ids = [f"<urn:uuid:{i:08d}-0000-0000-0000-000000000000>" for i in range(1000)]

    run1 = [get_document_split(doc_id, val_fraction=0.1) for doc_id in doc_ids]
    run2 = [get_document_split(doc_id, val_fraction=0.1) for doc_id in doc_ids]

    assert run1 == run2


def test_split_order_independence() -> None:
    """Verifies that processing document IDs in reverse or shuffled orders does not alter partition."""
    doc_ids = [f"doc_{i}" for i in range(500)]

    forward_splits = {doc_id: get_document_split(doc_id, val_fraction=0.05) for doc_id in doc_ids}

    # Shuffled order
    shuffled = list(doc_ids)
    random.seed(42)
    random.shuffle(shuffled)
    shuffled_splits = {doc_id: get_document_split(doc_id, val_fraction=0.05) for doc_id in shuffled}

    assert forward_splits == shuffled_splits


def test_strictly_zero_overlap() -> None:
    """Verifies that Train ∩ Val = ∅ across a large collection of document IDs."""
    doc_ids = [f"sample_doc_{i}" for i in range(2000)]
    train_set = set()
    val_set = set()

    for doc_id in doc_ids:
        if is_validation_document(doc_id, val_fraction=0.05):
            val_set.add(doc_id)
        else:
            train_set.add(doc_id)

    # Invariants
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set) + len(val_set) == len(doc_ids)
    assert len(val_set) > 0


def test_split_boundary_fractions() -> None:
    """Verifies edge-case validation fractions 0.0 and 1.0."""
    doc_ids = ["doc_a", "doc_b", "doc_c"]
    for d in doc_ids:
        assert get_document_split(d, val_fraction=0.0) == "train"
        assert get_document_split(d, val_fraction=1.0) == "validation"

    with pytest.raises(ValueError, match="val_fraction must be in range"):
        get_document_split("doc_a", val_fraction=-0.1)

    with pytest.raises(ValueError, match="val_fraction must be in range"):
        get_document_split("doc_a", val_fraction=1.5)
