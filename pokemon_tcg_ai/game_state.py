"""
game_state.py
--------------
Data structures representing a simplified Pokémon TCG battle state.

This is intentionally a lightweight, environment-agnostic representation.
It is designed to be easy to (a) simulate for synthetic dataset generation
and (b) map onto a real challenge environment's observation format by
writing a small adapter function (see `from_raw_observation`).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict


class EnergyType(str, Enum):
    FIRE = "Fire"
    WATER = "Water"
    GRASS = "Grass"
    LIGHTNING = "Lightning"
    PSYCHIC = "Psychic"
    FIGHTING = "Fighting"
    COLORLESS = "Colorless"


@dataclass
class Attack:
    name: str
    damage: int
    energy_cost: List[EnergyType]
    effect: Optional[str] = None  # e.g. "confuse", "heal_self_10", "discard_energy"


@dataclass
class Pokemon:
    name: str
    hp: int
    max_hp: int
    stage: int  # 0 = basic, 1 = stage1, 2 = stage2
    attacks: List[Attack] = field(default_factory=list)
    attached_energy: List[EnergyType] = field(default_factory=list)
    weakness: Optional[EnergyType] = None
    resistance: Optional[EnergyType] = None
    retreat_cost: int = 1
    status: Optional[str] = None  # "asleep", "paralyzed", "confused", "poisoned", "burned", None
    is_ex_or_gx: bool = False

    @property
    def is_fainted(self) -> bool:
        return self.hp <= 0

    @property
    def hp_ratio(self) -> float:
        return max(self.hp, 0) / self.max_hp if self.max_hp else 0.0


@dataclass
class PlayerBoard:
    active: Optional[Pokemon]
    bench: List[Pokemon] = field(default_factory=list)
    hand_size: int = 0
    deck_size: int = 40
    prize_cards_remaining: int = 6
    discard_pile_size: int = 0
    energy_in_hand: int = 0
    supporter_used_this_turn: bool = False

    @property
    def all_pokemon(self) -> List[Pokemon]:
        return ([self.active] if self.active else []) + self.bench


@dataclass
class GameState:
    turn_number: int
    is_my_turn: bool
    me: PlayerBoard
    opponent: PlayerBoard
    can_attach_energy: bool = True
    can_play_supporter: bool = True
    just_started_turn: bool = True

    def clone_summary(self) -> Dict:
        """Small serializable snapshot, useful for dataset rows / logging."""
        return {
            "turn_number": self.turn_number,
            "my_active_hp_ratio": self.me.active.hp_ratio if self.me.active else 0.0,
            "opp_active_hp_ratio": self.opponent.active.hp_ratio if self.opponent.active else 0.0,
            "my_bench_count": len(self.me.bench),
            "opp_bench_count": len(self.opponent.bench),
            "my_prizes_left": self.me.prize_cards_remaining,
            "opp_prizes_left": self.opponent.prize_cards_remaining,
        }


def from_raw_observation(obs: dict) -> GameState:
    """
    Adapter stub: convert a raw environment observation (dict/JSON, as
    typically provided by a Kaggle simulation harness) into a GameState.

    Replace the field-mapping below with the real competition schema once
    known. Keeping this isolated means the rest of the codebase never has
    to change when the raw format changes.
    """
    raise NotImplementedError(
        "Map the competition's raw observation schema to GameState here."
    )

