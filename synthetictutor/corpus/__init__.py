"""
Synthetictutor Corpus Package.
Tools for grounding corpus coverage matrix auditing, thin-cell enrichment, and manifest maintenance.
"""

from .coverage_audit import CorpusCoverageAuditor, CoverageStatus, ChapterCoverageRecord
from .chunk_enricher import CorpusChunkEnricher

__all__ = [
    "CorpusCoverageAuditor",
    "CoverageStatus",
    "ChapterCoverageRecord",
    "CorpusChunkEnricher",
]
