# Pokémon TCG AI Battle Challenge - Strategy Agent

## Project Overview
This project explores the use of artificial intelligence and strategic
decision-making to tackle the **Pokémon Trading Card Game AI Battle
Challenge**. The goal is an agent that analyzes the current game state
and selects effective actions during a battle, adapting to different
situations rather than relying on a fixed strategy.

## Repository Structure
```
pokemon-tcg-ai-battle/
├── pokemon_tcg_ai/
│   ├── game_state.py        # GameState / Pokemon / PlayerBoard data model
│   ├── actions.py            # Action space + legal_actions(state)
│   ├── evaluator.py          # Heuristic scoring of states and actions
│   ├── agent.py               # HeuristicAgent + MLAgent decision policies
│   ├── features.py            # (state, action) -> numeric feature vector
│   ├── dataset_generator.py   # Self-play simulator -> synthetic dataset
│   └── train_model.py         # Baseline ML model trained on the dataset
├── demo.py                    # End-to-end usage example
├── requirements.txt
└── README.md
```

## Approach
The workflow mirrors the strategy outlined for the challenge:

1. **Analyze the current game state** - `GameState` captures active/bench
   Pokémon, HP, energy, prizes, hand size, and turn context.
2. **Identify all available actions** - `legal_actions()` enumerates
   attacks (with an energy-cost check), retreats, energy attachment,
   supporter plays, and bench development.
3. **Evaluate the potential value and consequences of each action** -
   `evaluate_state()` scores the overall board; `score_action()` scores
   each candidate move (damage output, knockout/lethal bonuses, risk of
   over-extending, tempo/setup value).
4. **Select a strategy based on the current situation** -
   `HeuristicAgent.choose_action()` ranks all legal actions and picks the
   best-scoring one; `explain_decision()` prints the reasoning.
5. **Continuously adapt** - the same pipeline runs every turn against the
   live state, so decisions naturally change as HP, energy, prizes, and
   board presence evolve.

## Strategy Details
`evaluator.py` combines multiple weighted factors into a single score:

* Pokémon HP and health ratio (mine vs. opponent's)
* Attack damage, weakness/resistance modifiers, and knockout/lethal bonus
* Energy and resource development (rewarded more heavily early game)
* Board positioning (bench size differential)
* Risk penalty for attacking while critically low on HP
* Prize-card differential (long-term win condition, weighted highest)

This lets the agent balance **short-term** payoff (e.g., taking a
knockout now) against **long-term** positioning (e.g., developing the
bench and energy curve before committing to an all-in attack).

## Dataset
Since the real competition environment/rules engine may not be wired up
yet, `dataset_generator.py` ships a **simplified self-play simulator**
that produces a structurally realistic dataset:

* Random-but-plausible Pokémon (HP, attacks, weaknesses, energy costs)
  for both sides.
* An exploratory `HeuristicAgent` (with a configurable random-action
  rate) plays full games turn by turn.
* Each row logs the featurized `(state, action)` pair, the immediate
  reward, the change in heuristic board evaluation, and the eventual
  game outcome.

Run:
```bash
python -m pokemon_tcg_ai.dataset_generator
```
This produces `pokemon_tcg_battle_dataset.csv` (~6–7k rows from 500
simulated games by default).

To plug in the **real** competition environment instead of the
simplified simulator, implement `game_state.from_raw_observation()` and
replace `_apply_action_simplified()` in `dataset_generator.py` with calls
into the actual environment's step function — the agent, evaluator, and
feature code do not need to change.

## Baseline Model
`train_model.py` fits a `GradientBoostingRegressor` on the generated
dataset to predict action value directly from features, as a first
learnable step beyond the hand-tuned heuristic:
```bash
python -m pokemon_tcg_ai.train_model
```
On the bundled synthetic dataset this reaches **R² ≈ 0.80** on held-out
data, with `MLAgent` able to load the trained model and use it in place
of the heuristic scorer.

## Quick Start
```bash
pip install -r requirements.txt
python demo.py
```
`demo.py` builds a sample battle state, prints the agent's ranked
candidate actions and chosen move, then generates a small sample
dataset.

## Challenges
* **Dynamic action space** - the number of legal actions changes turn to
  turn (energy attached, bench size, hand contents), so a fixed strategy
  or fixed-size policy doesn't work; `legal_actions()` re-derives the
  space fresh every turn.
* **Short-term vs. long-term trade-offs** - an immediately-strong action
  (e.g., attacking now) can be worse than developing the board first;
  the weighted evaluator and the turn-based setup bonus address this
  directly.

## Results and Future Improvements
The heuristic agent produces sensible, explainable decisions (see
`demo.py` output) and the self-play simulator produces a usable labeled
dataset end-to-end. Natural next steps:

* Replace the simplified simulator with the real competition rules
  engine via `from_raw_observation()`.
* Add multi-ply lookahead / search (minimax or MCTS) on top of
  `evaluate_state()`.
* Move from the supervised baseline to true reinforcement learning
  (self-play with policy/value networks).
* Add opponent modeling (predict likely opponent actions rather than the
  current random opponent-turn stand-in).
* Large-scale self-play to refine the `WEIGHTS` in `evaluator.py` or to
  supersede them entirely with a learned value function.

## Conclusion
This project turns raw game-state information into ranked, explainable
strategic decisions, and adapts those decisions as a battle progresses.
It also ships a synthetic dataset and training pipeline so the same
codebase can grow from a rule-based baseline into a learned agent as
more game data becomes available.
# pokemon-tcg-ai-battle
# pokemon-tcg-ai-battle
