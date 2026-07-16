"""
Unified data augmentation interface.
Supports: EDA (synonym/swap/insert/delete) and Back-Translation.
"""

from .eda import augment_dataset as eda_augment
from .back_translate import augment_dataset_backtrans as bt_augment


def augment_dataset(texts, labels, method: str = "eda",
                    alpha: float = 0.1, num_aug: int = 1,
                    seed: int = 42):
    """
    Augment a dataset using the specified method.

    Args:
        texts: List of text strings
        labels: List of labels
        method: "eda" or "backtrans" or "both"
        alpha: Augmentation strength (for EDA)
        num_aug: Number of augmented copies per sample (for EDA)
        seed: Random seed

    Returns:
        (augmented_texts, augmented_labels)
    """
    if method == "eda":
        return eda_augment(texts, labels, alpha=alpha, num_aug=num_aug, seed=seed)
    elif method == "backtrans":
        return bt_augment(texts, labels, seed=seed)
    elif method == "both":
        # First apply EDA
        eda_texts, eda_labels = eda_augment(texts, labels, alpha=alpha, num_aug=num_aug, seed=seed)
        # Then back-translate (on the already EDA-augmented data)
        return bt_augment(eda_texts, eda_labels, seed=seed)
    else:
        raise ValueError(f"Unknown augmentation method: {method}")
