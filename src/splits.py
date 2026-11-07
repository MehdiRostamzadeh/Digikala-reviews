
"""Deterministic group-based dataset splitting."""

import pandas as pd


SPLIT_SEED = 2026

TRAIN_THRESHOLD = 7000
VALIDATION_THRESHOLD = 8500
HASH_KEY = "digikala2026seed"


def assign_splits(product_ids: pd.Series) -> pd.Series:
    """
    Assign products deterministically to train, validation, or test.

    Target proportions:
        Train:      70%
        Validation: 15%
        Test:       15%

    All rows belonging to the same product receive the same split.
    """

    if product_ids.isna().any():
        raise ValueError("product_ids contains missing values.")

    hashes = pd.util.hash_pandas_object(
        product_ids,
        index=False,
        hash_key=HASH_KEY,
    )

    buckets = hashes % 10_000

    splits = pd.Series(
        "test",
        index=product_ids.index,
        dtype="object",
    )

    splits.loc[buckets < TRAIN_THRESHOLD] = "train"

    splits.loc[
        (buckets >= TRAIN_THRESHOLD)
        & (buckets < VALIDATION_THRESHOLD)
    ] = "validation"

    return splits