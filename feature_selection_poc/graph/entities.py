from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureGraph:
    node_count: int
    edge_count: int
