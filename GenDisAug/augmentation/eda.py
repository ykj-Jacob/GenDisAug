"""
EDA: Easy Data Augmentation for Text Classification
Based on: Wei & Zou (EMNLP 2019) "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"

Four operations:
- SR: Synonym Replacement
- RI: Random Insertion
- RS: Random Swap
- RD: Random Deletion
"""

import random
import re
from typing import List

# Stopwords (common English words we avoid replacing)
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn',
])


def get_synonyms(word: str) -> List[str]:
    """Get synonyms from WordNet."""
    try:
        from nltk.corpus import wordnet
    except ImportError:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(text: str, n: int = 1) -> str:
    """Replace n random non-stopwords with synonyms."""
    words = text.split()
    if len(words) == 0:
        return text

    # Find replaceable words (non-stopwords, alphabetical)
    replaceable = [i for i, w in enumerate(words)
                   if w.lower() not in STOP_WORDS and w.isalpha()]

    if not replaceable:
        return text

    n = min(n, len(replaceable))
    indices = random.sample(replaceable, n)

    for idx in indices:
        synonyms = get_synonyms(words[idx])
        if synonyms:
            words[idx] = random.choice(synonyms)

    return ' '.join(words)


def random_insertion(text: str, n: int = 1) -> str:
    """Insert n random synonyms at random positions."""
    words = text.split()
    if len(words) == 0:
        return text

    # Find words that have synonyms
    candidates = [w for w in words if w.lower() not in STOP_WORDS and w.isalpha()]
    if not candidates:
        return text

    for _ in range(n):
        word = random.choice(candidates)
        synonyms = get_synonyms(word)
        if synonyms:
            new_word = random.choice(synonyms)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, new_word)

    return ' '.join(words)


def random_swap(text: str, n: int = 1) -> str:
    """Swap n random pairs of words."""
    words = text.split()
    if len(words) < 2:
        return text

    for _ in range(min(n, len(words) // 2)):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]

    return ' '.join(words)


def random_deletion(text: str, p: float = 0.1) -> str:
    """Delete each word with probability p, keeping at least 1 word."""
    words = text.split()
    if len(words) == 0:
        return text

    kept = [w for w in words if random.random() > p]
    if not kept:
        kept = [random.choice(words)]

    return ' '.join(kept)


def eda_augment(text: str, alpha_sr: float = 0.1, alpha_ri: float = 0.1,
                alpha_rs: float = 0.1, alpha_rd: float = 0.1,
                num_aug: int = 1) -> List[str]:
    """
    Apply EDA augmentation to a single text.
    Returns a list of augmented texts.

    alpha_*: percentage of words to modify for each operation
    num_aug: number of augmented samples to generate per operation
    """
    words = text.split()
    num_words = len(words)
    if num_words == 0:
        return [text]

    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    augmented = []
    for _ in range(num_aug):
        aug_text = text
        if random.random() < 0.5:
            aug_text = synonym_replacement(aug_text, n_sr)
        if random.random() < 0.5:
            aug_text = random_insertion(aug_text, n_ri)
        if random.random() < 0.5:
            aug_text = random_swap(aug_text, n_rs)
        if random.random() < 0.5:
            aug_text = random_deletion(aug_text, alpha_rd)
        augmented.append(aug_text)

    return augmented


def augment_dataset(texts, labels, alpha: float = 0.1, num_aug: int = 1,
                    seed: int = 42) -> tuple:
    """
    Augment an entire dataset with EDA.

    Args:
        texts: List of text strings
        labels: List of labels
        alpha: Augmentation strength (0.1 = 10% of words modified)
        num_aug: Number of augmented copies per original sample
        seed: Random seed

    Returns:
        (augmented_texts, augmented_labels) — original + augmented
    """
    random.seed(seed)
    import nltk
    try:
        from nltk.corpus import wordnet
    except (ImportError, LookupError):
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    aug_texts = list(texts)
    aug_labels = list(labels)

    for text, label in zip(texts, labels):
        augmented = eda_augment(text, alpha_sr=alpha, alpha_ri=alpha,
                                alpha_rs=alpha, alpha_rd=alpha, num_aug=num_aug)
        aug_texts.extend(augmented)
        aug_labels.extend([label] * len(augmented))

    # Shuffle
    combined = list(zip(aug_texts, aug_labels))
    random.shuffle(combined)
    aug_texts, aug_labels = zip(*combined)

    return list(aug_texts), list(aug_labels)


if __name__ == "__main__":
    # Quick test
    text = "the movie was incredibly good and I enjoyed it very much"
    print(f"Original: {text}")
    print(f"SR: {synonym_replacement(text, 2)}")
    print(f"RI: {random_insertion(text, 2)}")
    print(f"RS: {random_swap(text, 2)}")
    print(f"RD: {random_deletion(text, 0.2)}")
    print(f"EDA: {eda_augment(text, num_aug=2)}")
