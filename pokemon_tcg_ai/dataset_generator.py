"""
dataset_generator.py
---------------------
Generates a synthetic Pokémon-TCG-battle dataset via simplified self-play
simulation. This gives you something concrete to submit / train on for
the Kaggle challenge even before wiring up the real competition
environment: rows of (state features, action taken, resulting reward,
final game outcome).

The simulation is deliberately simplified (not a full rules engine) --
its purpose is to produce *structurally realistic* trajectories so the
feature pipeline, agent, and training code can all be validated
end-to-end. Swap `_simulate_one_game` internals for the real environment
once available.
"""

import csv
import random
from dataclasses import asdict
from typing import List, Dict

from .game_state import GameState, PlayerBoard, Pokemon, Attack, EnergyType
from .actions import legal_actions, ActionType, Action
from .agent import HeuristicAgent
from .evaluator import evaluate_state
from .features import featurize_state_action, FEATURE_NAMES


ENERGY_TYPES = [EnergyType.FIRE, EnergyType.WATER, EnergyType.GRASS,
                EnergyType.LIGHTNING, EnergyType.PSYCHIC, EnergyType.FIGHTING]


def _random_pokemon(rng: random.Random, name_prefix: str, idx: int) -> Pokemon:
    max_hp = rng.choice([60, 70, 80, 90, 100, 120, 140])
    etype = rng.choice(ENERGY_TYPES)
    weak = rng.choice(ENERGY_TYPES)
    atk1 = Attack(
        name="Quick Strike",
        damage=rng.choice([10, 20, 30]),
        energy_cost=[EnergyType.COLORLESS],
    )
    atk2 = Attack(
        name="Heavy Slam",
        damage=rng.choice([50, 60, 70, 80]),
        energy_cost=[etype, EnergyType.COLORLESS, EnergyType.COLORLESS],
    )
    return Pokemon(
        name=f"{name_prefix}{idx}",
        hp=max_hp,
        max_hp=max_hp,
        stage=rng.choice([0, 0, 1]),
        attacks=[atk1, atk2],
        attached_energy=[],
        weakness=weak,
        retreat_cost=rng.choice([1, 1, 2]),
    )


def _new_board(rng: random.Random, prefix: str) -> PlayerBoard:
    active = _random_pokemon(rng, prefix, 0)
    bench = [_random_pokemon(rng, prefix, i + 1) for i in range(rng.randint(0, 3))]
    return PlayerBoard(
        active=active,
        bench=bench,
        hand_size=rng.randint(2, 7),
        energy_in_hand=rng.randint(0, 3),
        prize_cards_remaining=6,
    )


def _apply_action_simplified(state: GameState, action: Action, rng: random.Random) -> float:
    """Apply an action with simplified effects and return an immediate
    scalar reward (used as a training signal). Mutates state in place."""
    me, opp = state.me, state.opponent
    reward = 0.0

    if action.action_type == ActionType.ATTACK and me.active and opp.active:
        attack = next((a for a in me.active.attacks if a.name == action.attack_name), None)
        if attack:
            dmg = attack.damage
            if opp.active.weakness and opp.active.weakness in attack.energy_cost:
                dmg *= 2
            opp.active.hp = max(0, opp.active.hp - dmg)
            reward += dmg / 20.0
            if opp.active.is_fainted:
                reward += 5.0
                me.prize_cards_remaining = max(0, me.prize_cards_remaining - 1)
                if opp.bench:
                    opp.active = opp.bench.pop(0)
                else:
                    opp.active = None

    elif action.action_type == ActionType.RETREAT and me.bench:
        me.active, me.bench[action.target_index] = me.bench[action.target_index], me.active
        reward -= 0.1

    elif action.action_type == ActionType.ATTACH_ENERGY:
        target = me.all_pokemon[action.target_index] if action.target_index is not None else me.active
        if target and me.energy_in_hand > 0:
            target.attached_energy.append(rng.choice(ENERGY_TYPES))
            me.energy_in_hand -= 1
            reward += 0.3

    elif action.action_type == ActionType.PLAY_SUPPORTER:
        me.supporter_used_this_turn = True
        me.hand_size = max(0, me.hand_size - 1)
        reward += 0.4

    elif action.action_type == ActionType.PLAY_BASIC_TO_BENCH and me.hand_size > 0:
        me.bench.append(_random_pokemon(rng, "Bench", len(me.bench) + 10))
        me.hand_size -= 1
        reward += 0.2

    return reward


def _simulate_one_game(rng: random.Random, agent: HeuristicAgent, max_turns: int = 40) -> List[Dict]:
    me_board = _new_board(rng, "MyMon")
    opp_board = _new_board(rng, "OppMon")
    state = GameState(turn_number=1, is_my_turn=True, me=me_board, opponent=opp_board)

    rows = []
    winner = None

    for turn in range(1, max_turns + 1):
        state.turn_number = turn
        state.is_my_turn = True
        state.me.supporter_used_this_turn = False

        actions = legal_actions(state)
        if not actions or state.me.active is None:
            winner = "opponent"
            break

        chosen = agent.choose_action(state)
        feats = featurize_state_action(state, chosen)
        pre_score = evaluate_state(state)
        reward = _apply_action_simplified(state, chosen, rng)
        post_score = evaluate_state(state)

        rows.append({
            **{name: val for name, val in zip(FEATURE_NAMES, feats)},
            "action_type": chosen.action_type.value,
            "immediate_reward": reward,
            "state_value_delta": post_score - pre_score,
            "turn": turn,
        })

        if state.opponent.prize_cards_remaining <= 0 or state.opponent.active is None and not state.opponent.bench:
            winner = "me"
            break

        # Very simplified opponent turn: random legal-ish attack or pass.
        if state.opponent.active and state.me.active:
            dmg = rng.choice([0, 10, 20, 30, 40])
            state.me.active.hp = max(0, state.me.active.hp - dmg)
            if state.me.active.is_fainted:
                state.opponent.prize_cards_remaining = max(0, state.opponent.prize_cards_remaining - 1)
                if state.me.bench:
                    state.me.active = state.me.bench.pop(0)
                else:
                    winner = "opponent"
                    break

    if winner is None:
        winner = "draw"

    for r in rows:
        r["game_outcome"] = winner

    return rows


def generate_dataset(n_games: int = 500, seed: int = 42, exploration: float = 0.15) -> List[Dict]:
    """Generate a synthetic dataset by simulating `n_games` self-play
    games with an exploratory heuristic agent (exploration adds action
    diversity so the dataset isn't dominated by one policy)."""
    rng = random.Random(seed)
    agent = HeuristicAgent(exploration=exploration, seed=seed)
    all_rows: List[Dict] = []
    for g in range(n_games):
        game_rng = random.Random(rng.randint(0, 10_000_000))
        rows = _simulate_one_game(game_rng, agent)
        for r in rows:
            r["game_id"] = g
        all_rows.extend(rows)
    return all_rows


def save_dataset_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        raise ValueError("No rows to save.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    data = generate_dataset(n_games=500)
    save_dataset_csv(data, "pokemon_tcg_battle_dataset.csv")
    print(f"Generated {len(data)} rows across simulated games.")

