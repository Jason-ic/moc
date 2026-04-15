"""
TriviaQA dataset wrapper with difficulty labeling and repeat-pattern injection.

Difficulty heuristic (token-free, fast):
  1 – Easy:   short answer (≤3 words), short question (<15 words), no reasoning keywords
  2 – Medium: everything in between
  3 – Hard:   long answer (>10 words) OR long question (>30 words) OR reasoning keywords
"""

import random
from collections import Counter
from torch.utils.data import Dataset
from datasets import load_dataset


REASONING_KEYWORDS = {"why", "how", "explain", "compare", "describe", "analyse",
                      "analyze", "relationship", "difference", "impact", "role"}


def estimate_difficulty(question: str, answer: str) -> int:
    q_words = question.lower().split()
    a_words = answer.split()
    q_len = len(q_words)
    a_len = len(a_words)
    has_complex = bool(set(q_words) & REASONING_KEYWORDS)

    if a_len <= 3 and q_len < 15 and not has_complex:
        return 1   # Easy
    if has_complex or a_len > 10 or q_len > 30:
        return 3   # Hard
    return 2       # Medium


def build_prompt(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


class TriviaQADataset(Dataset):
    """
    Loads TriviaQA and optionally injects repeated medium-difficulty patterns.

    Parameters
    ----------
    split              : 'train' | 'validation'
    tokenizer          : HuggingFace tokenizer
    max_len            : maximum token length (sequences are truncated / padded)
    repeat_fraction    : fraction of total samples that should be repeats
    n_repeat_patterns  : how many unique medium patterns to repeat
    max_samples        : cap dataset size (None = use all)
    seed               : random seed for reproducibility
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        max_len: int = 128,
        repeat_fraction: float = 0.3,
        n_repeat_patterns: int = 50,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.max_len = max_len

        raw = load_dataset("trivia_qa", "unfiltered.nocontext", split=split)

        texts: list[str] = []
        difficulties: list[int] = []

        for item in raw:
            q = item["question"]
            a = item["answer"]["value"]
            texts.append(build_prompt(q, a))
            difficulties.append(estimate_difficulty(q, a))

        if max_samples:
            texts = texts[:max_samples]
            difficulties = difficulties[:max_samples]

        # ------------------------------------------------------------------
        # Inject repeated medium-difficulty patterns
        # ------------------------------------------------------------------
        if split == "train" and repeat_fraction > 0 and n_repeat_patterns > 0:
            medium_idx = [i for i, d in enumerate(difficulties) if d == 2]
            n_pat = min(n_repeat_patterns, len(medium_idx))
            repeat_base = rng.sample(medium_idx, n_pat)
            self.repeat_indices = set(repeat_base)  # track for evaluation

            n_inject = int(len(texts) * repeat_fraction)
            for _ in range(n_inject):
                src = rng.choice(repeat_base)
                texts.append(texts[src])
                difficulties.append(difficulties[src])
        else:
            self.repeat_indices = set()

        self.texts = texts
        self.difficulties = difficulties

        # Quick stats
        cnt = Counter(difficulties)
        print(f"[TriviaQA {split}] total={len(texts)} | "
              f"easy={cnt[1]} med={cnt[2]} hard={cnt[3]} "
              f"repeats_injected={len(texts)-len(difficulties)+len(self.repeat_indices)}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        difficulty = self.difficulties[idx]

        if self.tokenizer is None:
            return {"text": text, "difficulty": difficulty}

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)         # (T,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (T,)

        # Labels: same as input_ids; ignore padding tokens in loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "difficulty": difficulty,
        }

    def difficulty_distribution(self) -> dict:
        cnt = Counter(self.difficulties)
        total = len(self.difficulties)
        return {d: cnt[d] / total for d in sorted(cnt)}
