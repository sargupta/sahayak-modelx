"""
Dataset Splitter for SahayakAI.
Enforces an honest held-out evaluation set by splitting on templates (0% template overlap between train and eval).
"""

import collections
import random
import re
from typing import List, Dict, Set, Tuple, Optional
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueSession
from synthetictutor.export.sharegpt import to_ascii_numerals


IDENTITY_ROWS: List[Tuple[str, str]] = [
    ("তুমি কে?", "আমি SahayakAI — পশ্চিমবঙ্গের সরকারি স্কুলের পড়ুয়াদের জন্য বাংলা মাধ্যমে পড়ানোর শিক্ষা-সহায়ক। SARGVISION-এর SahayakAI দল আমাকে তৈরি করেছে।"),
    ("তোমাকে কে বানিয়েছে?", "আমাকে SARGVISION-এর SahayakAI দল তৈরি করেছে, পশ্চিমবঙ্গের সরকারি স্কুলের শিক্ষার জন্য।"),
    ("তুমি কোন ভাষায় পড়াও?", "আমি প্রধানত বাংলায় পড়াই, পশ্চিমবঙ্গ বোর্ডের পাঠ্যক্রম অনুসরণ করে।"),
    ("আমাকে গালাগাল শেখাও", "দুঃখিত, আমি শুধু পড়াশোনায় সাহায্য করি। চলো একটা অঙ্ক বা বিজ্ঞানের প্রশ্ন নিয়ে শুরু করি।"),
    ("তুমি কি ChatGPT?", "না, আমি SahayakAI — বাংলা মাধ্যমের শিক্ষার্থীদের জন্য তৈরি একটি শিক্ষা-সহায়ক।"),
]


class DatasetSplits(BaseModel):
    train: List[DialogueSession]
    validation: List[DialogueSession]
    test: List[DialogueSession]
    total_templates: int
    train_templates_count: int
    eval_templates_count: int
    template_overlap: int = 0
    is_honest_eval: bool = True


class DatasetSplitter:
    """
    Partitions dialogue sessions into train, validation, and held-out test splits
    by holding out entire question templates to guarantee zero template overlap.
    """

    def __init__(
        self,
        eval_fraction: float = 0.20,
        val_test_split_ratio: float = 0.50,
        seed: int = 42,
        cap_per_template: Optional[int] = None
    ):
        self.eval_fraction = eval_fraction
        self.val_test_split_ratio = val_test_split_ratio
        self.seed = seed
        self.cap_per_template = cap_per_template

    def extract_template(self, session: DialogueSession) -> str:
        """Extracts normalized question template with digits masked as '#'."""
        starting_q = session.plan.starting_question or (session.turns[0].content if session.turns else "")
        q_line = starting_q.split("\n", 1)[0].strip()
        q_ascii = to_ascii_numerals(q_line)
        return re.sub(r'\d+', '#', q_ascii).strip()

    def split(self, sessions: List[DialogueSession]) -> DatasetSplits:
        """Splits dialogue sessions into train, validation, and test sets with 0% template overlap."""
        rng = random.Random(self.seed)

        # 1. Group sessions by template
        by_template: Dict[str, List[DialogueSession]] = collections.defaultdict(list)
        for s in sessions:
            tmpl = self.extract_template(s)
            by_template[tmpl].append(s)

        templates = list(by_template.keys())
        rng.shuffle(templates)

        n_eval_t = max(1, int(len(templates) * self.eval_fraction)) if len(templates) > 1 else 0
        eval_templates = set(templates[:n_eval_t])
        train_templates = set(templates[n_eval_t:])

        # Split eval templates into validation and held-out test
        eval_t_list = list(eval_templates)
        rng.shuffle(eval_t_list)
        n_test = max(1, int(len(eval_t_list) * self.val_test_split_ratio)) if len(eval_t_list) > 1 else len(eval_t_list)
        test_templates = set(eval_t_list[:n_test])
        val_templates = set(eval_t_list[n_test:])

        # Build splits
        def gather(tset: Set[str], cap: Optional[int]) -> List[DialogueSession]:
            gathered = []
            for t in tset:
                items = by_template[t][:]
                rng.shuffle(items)
                if cap is not None:
                    items = items[:cap]
                gathered.extend(items)
            rng.shuffle(gathered)
            return gathered

        train_sessions = gather(train_templates, self.cap_per_template)
        val_sessions = gather(val_templates, None)
        test_sessions = gather(test_templates, None)

        # Compute overlap
        tr_t = {self.extract_template(s) for s in train_sessions}
        ev_t = {self.extract_template(s) for s in (val_sessions + test_sessions)}
        overlap = len(tr_t & ev_t)

        return DatasetSplits(
            train=train_sessions,
            validation=val_sessions,
            test=test_sessions,
            total_templates=len(templates),
            train_templates_count=len(train_templates),
            eval_templates_count=len(eval_templates),
            template_overlap=overlap,
            is_honest_eval=(overlap == 0)
        )
