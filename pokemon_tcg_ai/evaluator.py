"""
evaluator.py
------------
Heuristic evaluation functions that score a GameState (board position)
and score individual Actions given a state. These implement the
"strategy" described in the project overview: prioritize decisions
that improve overall strategic position, not just immediate value.
"""

from typing import Dict
from .game_state import GameState, Pokemon
from .actions import Action, ActionType


# Tunable weights -- these are the "knobs" of the heuristic strategy.
WEIGHTS = {
    "hp_diff": 1.0,          # value of HP advantage
    "prize_diff": 3.0,       # value of being ahead on prizes
    "bench_diff": 0.5,       # value of board presence
    "energy_diff": 0.3,      # value of energy development
    "ko_bonus": 4.0,         # bonus for a move that knocks out a Pokémon
    "lethal_bonus": 20.0,    # bonus for a move that wins the game
    "risk_penalty": 1.5,     # penalty for leaving active Pokémon low HP into opp turn
    "setup_bonus": 0.4,      # bonus for developing board (bench/energy) early
}


def evaluate_state(state: GameState) -> float:
    """Return a scalar score: higher is better for `state.me`."""
    me, opp = state.me, state.opponent

    my_hp = sum(p.hp for p in me.all_pokemon)
    opp_hp = sum(p.hp for p in opp.all_pokemon)
    hp_diff = (my_hp - opp_hp) / 100.0

    # prize_diff > 0 means I have taken more prizes than the opponent
    prize_diff = (6 - opp.prize_cards_remaining) - (6 - me.prize_cards_remaining)

    bench_diff = len(me.bench) - len(opp.bench)

    my_energy = sum(len(p.attached_energy) for p in me.all_pokemon)
    opp_energy = sum(len(p.attached_energy) for p in opp.all_pokemon)
    energy_diff = my_energy - opp_energy

    score = (
        WEIGHTS["hp_diff"] * hp_diff +
        WEIGHTS["prize_diff"] * prize_diff +
        WEIGHTS["bench_diff"] * bench_diff +
        WEIGHTS["energy_diff"] * energy_diff
    )

    # Early game: reward board development.
    if state.turn_number <= 4:
        score += WEIGHTS["setup_bonus"] * (len(me.bench) + my_energy)

    return score


def score_action(state: GameState, action: Action) -> float:
    """
    Estimate the value of taking `action` from `state`.

    This does not do a full simulation (that is left to the optional
    lookahead / self-play components). Instead it applies fast, targeted
    heuristics per action type -- cheap enough to rank every legal action
    every turn.
    """
    me, opp = state.me, state.opponent
    base = 0.0

    if action.action_type == ActionType.ATTACK:
        attack = _find_attack(me.active, action.attack_name) if me.active else None
        if attack and opp.active:
            dmg = _effective_damage(attack, opp.active)
            base += dmg / 10.0

            # Bonus for a knockout.
            if dmg >= opp.active.hp:
                base += WEIGHTS["ko_bonus"]
                # Extra bonus if this would be the game-winning prize.
                if opp.prize_cards_remaining <= 1:
                    base += WEIGHTS["lethal_bonus"]

            # Penalize leaving ourselves exposed if opponent can KO back
            # on their next turn (rough proxy: our active is already low).
            if me.active.hp_ratio < 0.35:
                base -= WEIGHTS["risk_penalty"]

    elif action.action_type == ActionType.RETREAT:
        # Retreating a heavily damaged active for a healthy bench Pokémon
        # is good; retreating a healthy one wastes energy/tempo.
        if me.active:
            if me.active.hp_ratio < 0.3:
                base += 1.5
            else:
                base -= 0.5

    elif action.action_type == ActionType.ATTACH_ENERGY:
        base += 0.8  # generally good, enables future attacks
        if state.turn_number <= 3:
            base += 0.3  # extra value for early curve development

    elif action.action_type == ActionType.PLAY_SUPPORTER:
        base += 0.9  # card advantage / draw power is usually strong

    elif action.action_type == ActionType.PLAY_BASIC_TO_BENCH:
        base += 0.6 if len(me.bench) < 3 else 0.2  # diminishing returns

    elif action.action_type == ActionType.PASS_TURN:
        base -= 0.1  # slight penalty: avoid passing when better options exist

    return base


def _find_attack(pokemon: Pokemon, attack_name: str):
    if not pokemon:
        return None
    for a in pokemon.attacks:
        if a.name == attack_name:
            return a
    return None


def _effective_damage(attack, defender: Pokemon) -> int:
    dmg = attack.damage
    if defender.weakness and defender.weakness in attack.energy_cost:
        dmg *= 2
    if defender.resistance and defender.resistance in attack.energy_cost:
        dmg = max(0, dmg - 20)
    return dmg

