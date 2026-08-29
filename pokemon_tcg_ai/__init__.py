"""
pokemon_tcg_ai
--------------
Strategy agent for the Pokémon TCG AI Battle Challenge.

This file re-exports the main building blocks so you can do:

    from pokemon_tcg_ai import GameState, HeuristicAgent, evaluate_state

instead of importing from each submodule individually.
"""

from .game_state import GameState, PlayerBoard, Pokemon, Attack, EnergyType
from .actions import Action, ActionType, legal_actions
from .evaluator import evaluate_state, score_action
from .agent import HeuristicAgent, MLAgent, explain_decision
from .features import featurize_state_action, FEATURE_NAMES

__all__ = [
    "GameState", "PlayerBoard", "Pokemon", "Attack", "EnergyType",
    "Action", "ActionType", "legal_actions",
    "evaluate_state", "score_action",
    "HeuristicAgent", "MLAgent", "explain_decision",
    "featurize_state_action", "FEATURE_NAMES",
]