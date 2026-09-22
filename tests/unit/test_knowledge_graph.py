"""
Unit tests for KnowledgeGraph DAG builder, topological sort, and cycle validation.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from synthetictutor.core.schemas import Concept
from synthetictutor.knowledge.graph import KnowledgeGraph
from synthetictutor.core.exceptions import GraphValidationError


class TestKnowledgeGraph(unittest.TestCase):
    def test_knowledge_graph_topological_sort(self):
        c1 = Concept(id="c1", name="Force", description="Push or pull", prerequisite_ids=[])
        c2 = Concept(id="c2", name="Mass", description="Quantity of matter", prerequisite_ids=[])
        c3 = Concept(id="c3", name="Acceleration", description="Rate of change of velocity", prerequisite_ids=[])
        c4 = Concept(id="c4", name="Gravity Force", description="Attraction force", prerequisite_ids=["c1", "c2", "c3"])

        kg = KnowledgeGraph()
        kg.add_concept(c1)
        kg.add_concept(c2)
        kg.add_concept(c3)
        kg.add_concept(c4)

        teaching_order = kg.get_topological_teaching_order()
        ids = [c.id for c in teaching_order]

        self.assertLess(ids.index("c1"), ids.index("c4"))
        self.assertLess(ids.index("c2"), ids.index("c4"))
        self.assertLess(ids.index("c3"), ids.index("c4"))

    def test_knowledge_graph_cycle_detection(self):
        c1 = Concept(id="c1", name="A", description="A", prerequisite_ids=["c2"])
        c2 = Concept(id="c2", name="B", description="B", prerequisite_ids=["c1"])

        kg = KnowledgeGraph()
        kg.add_concept(c1)
        kg.add_concept(c2)

        with self.assertRaises(GraphValidationError):
            kg.get_topological_teaching_order()


if __name__ == "__main__":
    unittest.main()
