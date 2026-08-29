"""
actions.py
----------
Defines the space of possible actions in a turn and a function that
enumerates which actions are currently legal given a GameState.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .game_state import GameState, Pokemon, EnergyType


class ActionType(str, Enum):
    ATTACK = "ATTACK"
    RETREAT = "RETREAT"
    ATTACH_ENERGY = "ATTACH_ENERGY"
    PLAY_SUPPORTER = "PLAY_SUPPORTER"
    PLAY_BASIC_TO_BENCH = "PLAY_BASIC_TO_BENCH"
    EVOLVE = "EVOLVE"
    USE_ABILITY = "USE_ABILITY"
    PASS_TURN = "PASS_TURN"


@dataclass
class Action:
    action_type: ActionType
    source_index: Optional[int] = None   # index into me.all_pokemon
    target_index: Optional[int] = None   # index into opponent.all_pokemon or me.bench
    attack_name: Optional[str] = None
    energy_type: Optional[EnergyType] = None
    detail: Optional[str] = None

    def __repr__(self):
        parts = [self.action_type.value]
        if self.attack_name:
            parts.append(f"attack={self.attack_name}")
        if self.energy_type:
            parts.append(f"energy={self.energy_type.value}")
        if self.source_index is not None:
            parts.append(f"src={self.source_index}")
        if self.target_index is not None:
            parts.append(f"tgt={self.target_index}")
        return "Action(" + ", ".join(parts) + ")"


def legal_actions(state: GameState) -> List[Action]:
    """Enumerate all legal actions for the player-to-move."""
    if not state.is_my_turn:
        return []

    actions: List[Action] = [Action(ActionType.PASS_TURN)]
    me = state.me

    # Attacks from the active Pokémon, if enough energy attached.
    if me.active and not me.active.status == "asleep" and not me.active.status == "paralyzed":
        for atk in me.active.attacks:
            if _has_enough_energy(me.active.attached_energy, atk.energy_cost):
                actions.append(Action(ActionType.ATTACK, source_index=0, attack_name=atk.name))

    # Retreat: swap active with a healthy benched Pokémon.
    if me.active and len(me.bench) > 0:
        for i, benched in enumerate(me.bench):
            if not benched.is_fainted:
                actions.append(Action(ActionType.RETREAT, source_index=0, target_index=i))

    # Attach energy (assume at least one energy card available in hand
    # is tracked via me.energy_in_hand > 0; target = active or bench slot).
    if state.can_attach_energy and me.energy_in_hand > 0:
        for i, p in enumerate(me.all_pokemon):
            if not p.is_fainted:
                actions.append(Action(ActionType.ATTACH_ENERGY, target_index=i,
                                       energy_type=EnergyType.COLORLESS))

    # Play a supporter card once per turn.
    if state.can_play_supporter and not me.supporter_used_this_turn and me.hand_size > 0:
        actions.append(Action(ActionType.PLAY_SUPPORTER))

    # Play a basic Pokémon to an open bench slot.
    if len(me.bench) < 5 and me.hand_size > 0:
        actions.append(Action(ActionType.PLAY_BASIC_TO_BENCH))

    return actions


def _has_enough_energy(attached: List[EnergyType], cost: List[EnergyType]) -> bool:
    """Very simplified energy check: colorless cost can be paid by any
    energy; typed cost must be matched by count (ignoring dual-type
    edge cases for this simplified model)."""
    remaining = list(attached)
    typed_cost = [c for c in cost if c != EnergyType.COLORLESS]
    colorless_cost = len(cost) - len(typed_cost)

    for need in typed_cost:
        if need in remaining:
            remaining.remove(need)
        else:
            return False

    return len(remaining) >= colorless_cost

