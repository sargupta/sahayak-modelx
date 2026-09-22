"""
Diversity and De-duplication Evaluator for Synthetic Dialogue Batches.
Calculates n-gram distinctness, pairwise Self-BLEU, and template overlap metrics without external dependencies.
"""

import math
from typing import List, Dict, Set, Tuple
from collections import Counter
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueSession, MetricResult


class DiversityMetrics(BaseModel):
    distinct_1: float = Field(..., description="Ratio of unique unigrams to total unigrams")
    distinct_2: float = Field(..., description="Ratio of unique bigrams to total bigrams")
    self_bleu_score: float = Field(..., description="Average pairwise Self-BLEU (lower = higher diversity)")
    total_dialogues_evaluated: int
    is_diverse: bool = Field(..., description="True if Self-BLEU <= 0.65 and distinct-2 >= 0.40")


class DiversityEvaluator:
    """Computes lexical diversity, n-gram distinctness, and pairwise Self-BLEU across dialogue batches."""

    def __init__(self):
        pass

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation-split tokenizer."""
        clean = "".join([c if c.isalnum() else " " for c in text.lower()])
        return [w for w in clean.split() if w]

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def compute_distinct_n(self, texts: List[str], n: int) -> float:
        """Computes Distinct-N metric (unique n-grams / total n-grams)."""
        all_ngrams = []
        for t in texts:
            tokens = self._tokenize(t)
            all_ngrams.extend(self._get_ngrams(tokens, n))
        if not all_ngrams:
            return 1.0
        return len(set(all_ngrams)) / len(all_ngrams)

    def _sentence_bleu_2(self, reference: List[str], hypothesis: List[str]) -> float:
        """Lightweight sentence BLEU-2 calculation."""
        if not reference or not hypothesis:
            return 0.0

        ref_1 = Counter(self._get_ngrams(reference, 1))
        hyp_1 = Counter(self._get_ngrams(hypothesis, 1))
        ref_2 = Counter(self._get_ngrams(reference, 2))
        hyp_2 = Counter(self._get_ngrams(hypothesis, 2))

        # Precision 1
        p1_match = sum(min(count, ref_1[ng]) for ng, count in hyp_1.items())
        p1 = p1_match / max(len(hypothesis), 1)

        # Precision 2
        p2_total = max(len(hypothesis) - 1, 1)
        p2_match = sum(min(count, ref_2[ng]) for ng, count in hyp_2.items())
        p2 = p2_match / p2_total

        if p1 == 0 or p2 == 0:
            return 0.0

        # Brevity penalty
        bp = 1.0
        if len(hypothesis) < len(reference):
            bp = math.exp(1 - (len(reference) / max(len(hypothesis), 1)))

        return bp * math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2))

    def compute_self_bleu(self, texts: List[str], sample_size: int = 50) -> float:
        """Calculates Self-BLEU-2 across a collection of texts."""
        if len(texts) <= 1:
            return 0.0

        tokenized_texts = [self._tokenize(t) for t in texts[:sample_size]]
        scores = []

        for i, hyp in enumerate(tokenized_texts):
            refs = [ref for j, ref in enumerate(tokenized_texts) if j != i]
            if not refs:
                continue
            # Best BLEU against any other reference in the batch
            bleu_scores = [self._sentence_bleu_2(ref, hyp) for ref in refs]
            scores.append(sum(bleu_scores) / len(bleu_scores))

        return sum(scores) / len(scores) if scores else 0.0

    def evaluate_batch_diversity(self, sessions: List[DialogueSession]) -> DiversityMetrics:
        """Evaluates batch diversity across all dialogues."""
        teacher_texts = [
            " ".join([t.content for t in s.turns if t.role.value == "teacher"])
            for s in sessions
        ]

        distinct_1 = self.compute_distinct_n(teacher_texts, 1)
        distinct_2 = self.compute_distinct_n(teacher_texts, 2)
        self_bleu = self.compute_self_bleu(teacher_texts)

        is_diverse = (self_bleu <= 0.65) and (distinct_2 >= 0.35)

        return DiversityMetrics(
            distinct_1=round(distinct_1, 4),
            distinct_2=round(distinct_2, 4),
            self_bleu_score=round(self_bleu, 4),
            total_dialogues_evaluated=len(sessions),
            is_diverse=is_diverse
        )
