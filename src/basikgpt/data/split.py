"""Deterministic document-level train/validation split using cryptographic hashing.

Assigns documents to 'train' or 'validation' splits based on stable document identifiers
and SHA-256 hashing, guaranteeing order-independent, reproducible partitioning
with strictly zero overlap (Train ∩ Val = ∅).
"""

import hashlib


def get_document_split(
    doc_id: str,
    val_fraction: float = 0.005,
    salt: str = "basikgpt-fineweb-edu-v1",
) -> str:
    """Determines whether a document belongs to 'train' or 'validation' partition.

    Uses SHA-256 cryptographic hashing to map document IDs uniformly into 10,000 buckets
    [0..9999]. Documents falling below `val_fraction * 10000` are assigned to validation.

    Properties:
        - Deterministic: Always produces identical assignments across runs and machines.
        - Order-independent: Processing documents in different orders yields identical splits.
        - Zero Overlap: Document ID partitioning guarantees Train ∩ Val = ∅.

    Args:
        doc_id: Unique string identifier for the document (e.g. URN UUID).
        val_fraction: Fraction of documents to allocate to validation (e.g. 0.005 = 0.5%).
        salt: Domain salt to prevent collision with other hashing schemes.

    Returns:
        "validation" if document falls into validation bucket, otherwise "train".
    """
    if not (0.0 <= val_fraction <= 1.0):
        raise ValueError(f"val_fraction must be in range [0.0, 1.0], got {val_fraction}")

    if val_fraction == 0.0:
        return "train"
    if val_fraction == 1.0:
        return "validation"

    # Compute stable SHA-256 digest over salt and document ID
    payload = f"{salt}:{doc_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    # Map first 8 hex characters (32-bit unsigned int) into bucket [0..9999]
    bucket = int(digest[:8], 16) % 10000
    threshold = int(val_fraction * 10000)

    return "validation" if bucket < threshold else "train"


def is_validation_document(
    doc_id: str,
    val_fraction: float = 0.005,
    salt: str = "basikgpt-fineweb-edu-v1",
) -> bool:
    """Convenience predicate returning True if document is in validation split."""
    return get_document_split(doc_id, val_fraction=val_fraction, salt=salt) == "validation"
