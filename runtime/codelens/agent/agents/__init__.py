"""Lazy public API for localization operators and data contracts.

Keeping this package initializer lazy prevents models -> package -> pipeline ->
policy -> models import cycles while preserving the former public names.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "RetrievalAgent",
    "LocalizeAgent",
    "CriticAgent",
    "ReflectAgent",
    "LocalizationPipeline",
    "CriticEvaluation",
    "GraphContext",
    "GraphEdge",
    "GraphNode",
    "GraphPath",
    "LocalizeResult",
    "ReflectionAction",
    "ReflectionDecision",
    "TargetLocation",
]


_PUBLIC_MODULES = {
    "RetrievalAgent": "agent.agents.retrieval_agent",
    "LocalizeAgent": "agent.agents.localize_agent",
    "CriticAgent": "agent.agents.critic_agent",
    "ReflectAgent": "agent.agents.reflect_agent",
    "LocalizationPipeline": "agent.agents.pipeline",
    "CriticEvaluation": "agent.agents.models",
    "GraphContext": "agent.agents.models",
    "GraphEdge": "agent.agents.models",
    "GraphNode": "agent.agents.models",
    "GraphPath": "agent.agents.models",
    "LocalizeResult": "agent.agents.models",
    "ReflectionAction": "agent.agents.models",
    "ReflectionDecision": "agent.agents.models",
    "TargetLocation": "agent.agents.models",
}


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
