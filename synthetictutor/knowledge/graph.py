"""
Knowledge Representation Module: Concept Directed Acyclic Graph (DAG).
Supports pure-Python topological sorting, cycle detection, and ancestor extraction
with optional NetworkX interoperability.
"""

from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, deque
from synthetictutor.core.schemas import Concept, ConceptGraphData
from synthetictutor.core.exceptions import GraphValidationError


class KnowledgeGraph:
    """Manages concept dependency DAG and prerequisite topological ordering."""

    def __init__(self):
        self._concepts: Dict[str, Concept] = {}
        # Adjacency list: prereq_id -> set of downstream concept_ids
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        # Reverse adjacency: concept_id -> set of upstream prereq_ids
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_concept(self, concept: Concept):
        """Adds a concept node and its prerequisite directed edges to the graph."""
        self._concepts[concept.id] = concept
        for prereq_id in concept.prerequisite_ids:
            self._adjacency[prereq_id].add(concept.id)
            self._reverse_adjacency[concept.id].add(prereq_id)

    def validate(self):
        """Validates that the concept graph is a valid Directed Acyclic Graph (DAG)."""
        # Ensure all referenced prerequisite nodes exist
        for concept_id, concept in self._concepts.items():
            for prereq_id in concept.prerequisite_ids:
                if prereq_id not in self._concepts:
                    raise GraphValidationError(
                        f"Concept '{concept.name}' ({concept_id}) references missing prerequisite concept ID: '{prereq_id}'"
                    )

        # Cycle detection using 3-color DFS
        visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting (in stack), 2: visited

        def has_cycle(node_id: str, path: List[str]) -> bool:
            visited[node_id] = 1
            path.append(node_id)
            for neighbor_id in self._adjacency.get(node_id, set()):
                if neighbor_id not in self._concepts:
                    continue
                if visited.get(neighbor_id, 0) == 1:
                    cycle = path[path.index(neighbor_id):] + [neighbor_id]
                    raise GraphValidationError(f"Cyclic dependency detected in concept graph: {cycle}")
                elif visited.get(neighbor_id, 0) == 0:
                    if has_cycle(neighbor_id, path):
                        return True
            path.pop()
            visited[node_id] = 2
            return False

        for cid in list(self._concepts.keys()):
            if visited.get(cid, 0) == 0:
                has_cycle(cid, [])

    def get_topological_teaching_order(self) -> List[Concept]:
        """Returns concepts ordered by prerequisite dependency (Kahn's topological sort)."""
        self.validate()

        in_degree: Dict[str, int] = {cid: 0 for cid in self._concepts}
        for cid, concept in self._concepts.items():
            for prereq_id in concept.prerequisite_ids:
                if prereq_id in self._concepts:
                    in_degree[cid] += 1

        queue = deque([cid for cid, deg in in_degree.items() if deg == 0])
        sorted_ids: List[str] = []

        while queue:
            node = queue.popleft()
            sorted_ids.append(node)
            for neighbor in self._adjacency.get(node, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        if len(sorted_ids) != len(self._concepts):
            raise GraphValidationError("Cyclic dependency encountered during topological sort.")

        return [self._concepts[cid] for cid in sorted_ids if cid in self._concepts]

    def get_prerequisites(self, concept_id: str) -> List[Concept]:
        """Returns direct and ancestor prerequisite concepts for a given concept."""
        if concept_id not in self._concepts:
            raise GraphValidationError(f"Concept ID '{concept_id}' not found in knowledge graph.")

        ancestors: Set[str] = set()
        queue = deque(self._reverse_adjacency.get(concept_id, set()))

        while queue:
            curr = queue.popleft()
            if curr in self._concepts and curr not in ancestors:
                ancestors.add(curr)
                queue.extend(self._reverse_adjacency.get(curr, set()))

        return [self._concepts[aid] for aid in ancestors if aid in self._concepts]

    def to_schema(self) -> ConceptGraphData:
        """Serializes KnowledgeGraph into Pydantic schema."""
        edges: List[Tuple[str, str]] = []
        for src, dests in self._adjacency.items():
            for dst in dests:
                if src in self._concepts and dst in self._concepts:
                    edges.append((src, dst))
        return ConceptGraphData(concepts=self._concepts, edges=edges)

    @classmethod
    def from_schema(cls, data: ConceptGraphData) -> "KnowledgeGraph":
        """Reconstructs KnowledgeGraph from Pydantic schema."""
        kg = cls()
        for concept in data.concepts.values():
            kg.add_concept(concept)
        kg.validate()
        return kg
