"""
agent.py
--------
The main decision-making agent. Implements the workflow described in the
project overview:

  1. Analyze the current game state.
  2. Identify all available actions.
  3. Evaluate the potential value and consequences of each action.
  4. Select a strategy based on the current situation.
  5. Continuously adapt decisions as the game evolves.

Two agents are provided:
  - HeuristicAgent: fast, explainable, rule/weight-based (default, no
    training required -- good baseline for the Kaggle challenge).
  - MLAgent: wraps a trained scikit-learn model (see train_model.py) that
    predicts action value from featurized (state, action) pairs, useful
    once you have logged self-play data.
"""

from typing import List, Optional
import random

from .game_state import GameState
from .actions import Action, ActionType, legal_actions
from .evaluator import evaluate_state, score_action
from .features import featurize_state_action


class HeuristicAgent:
    def __init__(self, exploration: float = 0.0, seed: Optional[int] = None):
        """
        exploration: probability of picking a random legal action instead
        of the top-scored one. Useful for generating diverse self-play
        data (see dataset_generator.py). Keep at 0.0 for competition play.
        """
        self.exploration = exploration
        self.rng = random.Random(seed)

    def choose_action(self, state: GameState) -> Action:
        actions = legal_actions(state)
        if not actions:
            return Action(ActionType.PASS_TURN)

        if self.exploration > 0 and self.rng.random() < self.exploration:
            return self.rng.choice(actions)

        scored = [(score_action(state, a), a) for a in actions]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def rank_actions(self, state: GameState) -> List[tuple]:
        """Return (score, action) pairs sorted best-first. Useful for
        debugging / explaining a decision."""
        actions = legal_actions(state)
        scored = [(score_action(state, a), a) for a in actions]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


class MLAgent:
    """Uses a trained regressor to predict a value for each (state, action)
    pair and picks the max. Falls back to HeuristicAgent if no model is
    loaded, so the pipeline always produces a legal move."""

    def __init__(self, model=None):
        self.model = model
        self._fallback = HeuristicAgent()

    def choose_action(self, state: GameState) -> Action:
        actions = legal_actions(state)
        if not actions:
            return Action(ActionType.PASS_TURN)

        if self.model is None:
            return self._fallback.choose_action(state)

        feats = [featurize_state_action(state, a) for a in actions]
        preds = self.model.predict(feats)
        best_idx = max(range(len(actions)), key=lambda i: preds[i])
        return actions[best_idx]


def explain_decision(state: GameState, agent: HeuristicAgent, top_k: int = 3) -> str:
    """Human-readable explanation of the top-k considered actions and why
    the winner was chosen -- handy for a Kaggle write-up / notebook demo."""
    ranked = agent.rank_actions(state)[:top_k]
    board_score = evaluate_state(state)
    lines = [f"Current board evaluation: {board_score:.2f}", "Top candidate actions:"]
    for score, action in ranked:
        lines.append(f"  {action}  ->  score={score:.2f}")
    return "\n".join(lines)
