"""
Back-Translation Data Augmentation
Uses MarianMT (Helsinki-NLP) for English → German → English translation.
"""

from typing import List
import torch

# Lazy-loaded models
_en_de_model = None
_en_de_tokenizer = None
_de_en_model = None
_de_en_tokenizer = None
_device = None


def _load_models():
    """Lazy load translation models — loads only what's missing."""
    global _en_de_model, _en_de_tokenizer, _de_en_model, _de_en_tokenizer, _device

    if _en_de_model is not None and _de_en_model is not None:
        return

    from transformers import MarianMTModel, MarianTokenizer

    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # English → German (skip if already loaded)
    if _en_de_model is None:
        print("Loading EN→DE translation model...")
        _en_de_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        _en_de_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-de", use_safetensors=True
        ).to(_device)
        _en_de_model.eval()

    # German → English (skip if already loaded)
    if _de_en_model is None:
        print("Loading DE→EN translation model...")
        _de_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        _de_en_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en", use_safetensors=True
        ).to(_device)
        _de_en_model.eval()

    print("Back-translation models loaded.")


def translate(texts: List[str], tokenizer, model, max_length: int = 256) -> List[str]:
    """Translate a batch of texts."""
    batch_size = 8
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            translated = model.generate(**inputs, max_length=max_length, num_beams=4,
                                        early_stopping=True)

        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)

    return results


def back_translate(texts: List[str]) -> List[str]:
    """
    Apply back-translation: English → German → English.

    Args:
        texts: List of English texts

    Returns:
        List of back-translated English texts
    """
    _load_models()

    # EN → DE
    german_texts = translate(texts, _en_de_tokenizer, _en_de_model)

    # DE → EN
    back_translated = translate(german_texts, _de_en_tokenizer, _de_en_model)

    return back_translated


def back_translate_single(text: str) -> str:
    """Back-translate a single text."""
    return back_translate([text])[0]


def augment_dataset_backtrans(texts, labels, seed: int = 42) -> tuple:
    """
    Augment a dataset with back-translation.
    Adds one back-translated copy per original sample (doubles dataset size).

    Args:
        texts: List of text strings
        labels: List of labels
        seed: Random seed

    Returns:
        (augmented_texts, augmented_labels) — original + back-translated
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Back-translating {len(texts)} texts...")
    bt_texts = back_translate(texts)

    # Combine original + back-translated
    all_texts = list(texts) + bt_texts
    all_labels = list(labels) + list(labels)

    # Shuffle
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)

    return list(all_texts), list(all_labels)


if __name__ == "__main__":
    # Quick test
    text = "the food was delicious and the service was excellent"
    print(f"Original: {text}")
    print(f"Back-translated: {back_translate_single(text)}")
