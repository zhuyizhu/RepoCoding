"""In-memory append-only audit records and local adjacency indices."""
from copy import deepcopy

class StateStore:
    def __init__(self):
        self.hypotheses = {}
        self.observations = {}
        self.evidence = {}
        self.bindings = {}
        self.refinements = []
        self.events = []
        self.state_changes = []
        self.errors = []
        self.transitions = {}
        self.evidence_validity = {}
        self.anchor_hypotheses = {}
        self.anchor_evidence = {}
        self.evidence_bindings = {}
        self.hypothesis_bindings = {}
        self.origin_evidence = {}
        self.children = {}
        self.event_hashes = {}
        self.event_deltas = {}

    def get_hypothesis(self, hypothesis_id, snapshot=None):
        return deepcopy((snapshot['hypotheses'] if snapshot else self.hypotheses)[hypothesis_id])
