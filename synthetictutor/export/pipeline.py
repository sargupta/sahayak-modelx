"""
Unified Dataset Export Pipeline for SahayakAI & Sarvam-30B QLoRA.
Orchestrates multi-split generation, multi-format export (ShareGPT, ChatML, Alpaca),
and dataset card generation with zero-overlap validation.
"""

import os
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueSession
from synthetictutor.export.sharegpt import ShareGPTExporter
from synthetictutor.export.huggingface import ChatMLExporter
from synthetictutor.export.alpaca import AlpacaExporter
from synthetictutor.export.splitter import DatasetSplitter, DatasetSplits
from synthetictutor.export.dataset_card import DatasetCardGenerator


class ExportManifest(BaseModel):
    output_directory: str
    train_count: int
    validation_count: int
    test_count: int
    template_overlap: int
    is_honest_eval: bool
    generated_files: Dict[str, str]


class DatasetExportPipeline:
    """
    End-to-end dataset export pipeline that converts evaluated DialogueSessions into
    fine-tuning datasets ready for Sarvam-30B QLoRA and Hugging Face Hub.
    """

    def __init__(
        self,
        eval_fraction: float = 0.20,
        val_test_split_ratio: float = 0.50,
        include_inner_thought: bool = True,
        numerals_mode: str = "bengali",
        seed: int = 42
    ):
        self.splitter = DatasetSplitter(
            eval_fraction=eval_fraction,
            val_test_split_ratio=val_test_split_ratio,
            seed=seed
        )
        self.sharegpt_exporter = ShareGPTExporter(
            include_inner_thought=include_inner_thought,
            numerals_mode=numerals_mode
        )
        self.chatml_exporter = ChatMLExporter(
            include_inner_thought=include_inner_thought,
            numerals_mode=numerals_mode
        )
        self.alpaca_exporter = AlpacaExporter(
            include_inner_thought=include_inner_thought,
            numerals_mode=numerals_mode
        )
        self.card_generator = DatasetCardGenerator()
        self.numerals_mode = numerals_mode

    def run(
        self,
        sessions: List[DialogueSession],
        output_dir: str
    ) -> ExportManifest:
        """
        Executes full splitting, formatting, and artifact generation.
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_files: Dict[str, str] = {}

        # 1. Split sessions on templates (zero-overlap)
        splits: DatasetSplits = self.splitter.split(sessions)

        # 2. Export ShareGPT format
        sharegpt_dir = os.path.join(output_dir, "sharegpt")
        generated_files["sharegpt_train"] = self.sharegpt_exporter.export(splits.train, os.path.join(sharegpt_dir, "train.jsonl"))
        generated_files["sharegpt_val"] = self.sharegpt_exporter.export(splits.validation, os.path.join(sharegpt_dir, "val.jsonl"))
        generated_files["sharegpt_test"] = self.sharegpt_exporter.export(splits.test, os.path.join(sharegpt_dir, "test.jsonl"))

        # 3. Export ChatML format
        chatml_dir = os.path.join(output_dir, "chatml")
        generated_files["chatml_train"] = self.chatml_exporter.export(splits.train, os.path.join(chatml_dir, "train.jsonl"))
        generated_files["chatml_val"] = self.chatml_exporter.export(splits.validation, os.path.join(chatml_dir, "val.jsonl"))
        generated_files["chatml_test"] = self.chatml_exporter.export(splits.test, os.path.join(chatml_dir, "test.jsonl"))

        # 4. Export Alpaca format
        alpaca_dir = os.path.join(output_dir, "alpaca")
        generated_files["alpaca_train"] = self.alpaca_exporter.export(splits.train, os.path.join(alpaca_dir, "train.jsonl"))
        generated_files["alpaca_val"] = self.alpaca_exporter.export(splits.validation, os.path.join(alpaca_dir, "val.jsonl"))
        generated_files["alpaca_test"] = self.alpaca_exporter.export(splits.test, os.path.join(alpaca_dir, "test.jsonl"))

        # 5. Write Dataset Card & Info JSON
        card_artifacts = self.card_generator.write_artifacts(
            splits=splits,
            output_dir=output_dir,
            numerals_mode=self.numerals_mode
        )
        generated_files.update(card_artifacts)

        return ExportManifest(
            output_directory=output_dir,
            train_count=len(splits.train),
            validation_count=len(splits.validation),
            test_count=len(splits.test),
            template_overlap=splits.template_overlap,
            is_honest_eval=splits.is_honest_eval,
            generated_files=generated_files
        )
