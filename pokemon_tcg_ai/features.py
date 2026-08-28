"""
features.py
-----------
Turns a (GameState, Action) pair into a fixed-length numeric feature
vector, for use by a trainable ML model (see train_model.py). Keeping
this separate from the heuristic evaluator makes it easy to swap in a
learned model without touching the rest of the agent code.
"""

from typing import List
from .game_state import GameState
from .actions import Action, ActionType

ACTION_TYPES = list(ActionType)

FEATURE_NAMES = [
    "turn_number",
    "my_active_hp_ratio",
    "opp_active_hp_ratio",
    "my_bench_count",
    "opp_bench_count",
    "my_prizes_left",
    "opp_prizes_left",
    "my_energy_in_hand",
    "my_hand_size",
] + [f"action_is_{t.value}" for t in ACTION_TYPES]


def featurize_state_action(state: GameState, action: Action) -> List[float]:
    me, opp = state.me, state.opponent
    base = [
        float(state.turn_number),
        me.active.hp_ratio if me.active else 0.0,
        opp.active.hp_ratio if opp.active else 0.0,
        float(len(me.bench)),
        float(len(opp.bench)),
        float(me.prize_cards_remaining),
        float(opp.prize_cards_remaining),
        float(me.energy_in_hand),
        float(me.hand_size),
    ]
    one_hot = [1.0 if action.action_type == t else 0.0 for t in ACTION_TYPES]
    return base + one_hot
