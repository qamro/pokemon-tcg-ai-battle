"""
demo.py
-------
Small end-to-end demo:
1. Build a sample GameState by hand.
2. Ask the HeuristicAgent to choose an action and explain its reasoning.
3. Generate a small synthetic dataset.
4. Train a baseline model on it.

Run: python demo.py
"""

from pokemon_tcg_ai.game_state import GameState, PlayerBoard, Pokemon, Attack, EnergyType
from pokemon_tcg_ai.agent import HeuristicAgent, explain_decision
from pokemon_tcg_ai.dataset_generator import generate_dataset, save_dataset_csv


def build_sample_state() -> GameState:
    my_active = Pokemon(
        name="Charizard",
        hp=80, max_hp=150, stage=2,
        attacks=[
            Attack("Ember", 30, [EnergyType.FIRE]),
            Attack("Fire Blast", 120, [EnergyType.FIRE, EnergyType.FIRE, EnergyType.COLORLESS]),
        ],
        attached_energy=[EnergyType.FIRE, EnergyType.FIRE, EnergyType.COLORLESS],
        weakness=EnergyType.WATER,
    )
    my_bench = [
        Pokemon("Pidgey", 60, 60, 0, attacks=[Attack("Peck", 10, [EnergyType.COLORLESS])]),
    ]
    opp_active = Pokemon(
        name="Blastoise",
        hp=40, max_hp=160, stage=2,
        attacks=[Attack("Hydro Pump", 90, [EnergyType.WATER, EnergyType.WATER])],
        attached_energy=[EnergyType.WATER, EnergyType.WATER],
        weakness=EnergyType.LIGHTNING,
    )

    me = PlayerBoard(active=my_active, bench=my_bench, hand_size=4, energy_in_hand=1, prize_cards_remaining=3)
    opp = PlayerBoard(active=opp_active, bench=[], hand_size=3, energy_in_hand=0, prize_cards_remaining=1)

    return GameState(turn_number=8, is_my_turn=True, me=me, opponent=opp)


def main():
    state = build_sample_state()
    agent = HeuristicAgent()

    print("=== Sample decision ===")
    print(explain_decision(state, agent))
    chosen = agent.choose_action(state)
    print(f"\nChosen action: {chosen}\n")

    print("=== Generating a small synthetic dataset (50 games) ===")
    rows = generate_dataset(n_games=50)
    save_dataset_csv(rows, "pokemon_tcg_battle_dataset_sample.csv")
    print(f"Wrote {len(rows)} rows to pokemon_tcg_battle_dataset_sample.csv")


if __name__ == "__main__":
    main()

