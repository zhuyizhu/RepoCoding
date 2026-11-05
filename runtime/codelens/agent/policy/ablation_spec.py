"""Frozen component switches for the CodeLens mechanism ablations.

The older named screens remain supported by the pipeline.  These switches
only describe the five mechanism ablations and the matched ``full`` path.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSpec:
    propagation: bool = True
    scope_recovery: bool = True
    relocalize: bool = True
    outside_mass: bool = True
    controller: str = "voi"


ABLATION_SPECS = {
    "full": AblationSpec(),
    "wo_c1": AblationSpec(propagation=False, scope_recovery=False, relocalize=False),
    "wo_c2": AblationSpec(outside_mass=False),
    "wo_c3": AblationSpec(controller="heuristic"),
    "wo_propagation": AblationSpec(propagation=False),
    "wo_scope_recovery": AblationSpec(scope_recovery=False),
}


def specification_for(variant: str) -> AblationSpec:
    """Older screen names intentionally retain their historical behavior."""
    return ABLATION_SPECS.get(variant, ABLATION_SPECS["full"])
