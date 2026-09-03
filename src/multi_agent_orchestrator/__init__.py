"""Multi-Agent Orchestrator — a self-extending meta-agent over sub-agents.

A meta-agent plans a task DAG, routes each step to a Web Agent or an API Agent,
runs a plan -> execute -> verify -> iterate loop that re-plans on failure, and
synthesizes new sub-agents at runtime when the plan needs a capability no
registered agent has. Wrapped in an audit / guardrail / cost-budget layer.

The whole system runs offline and deterministically with no API key; set
ANTHROPIC_API_KEY to swap the heuristic planner/verifier/synthesizer for a real
LLM. See README.md.
"""

from .types import AgentSpec, Observation, Plan, RunReport, Step, StepResult

__all__ = [
    "AgentSpec",
    "Observation",
    "Plan",
    "RunReport",
    "Step",
    "StepResult",
]

__version__ = "0.1.0"
