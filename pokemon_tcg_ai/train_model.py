"""
train_model.py
---------------
Trains a baseline regression model to predict action value
(state_value_delta + immediate_reward) from the featurized dataset
produced by dataset_generator.py. This is the "reinforcement-learning-
flavored supervised baseline" mentioned as a future improvement in the
project write-up: instead of full RL, we regress on logged self-play
outcomes as a first learnable step beyond the hand-tuned heuristic.

Usage:
    python -m pokemon_tcg_ai.train_model
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from .features import FEATURE_NAMES


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Target = immediate reward plus a discounted bonus if the game was
    eventually won, so actions in winning games are reinforced."""
    outcome_bonus = df["game_outcome"].map({"me": 2.0, "opponent": -2.0, "draw": 0.0}).fillna(0.0)
    df = df.copy()
    df["target"] = df["immediate_reward"] + df["state_value_delta"] + 0.1 * outcome_bonus
    return df


def train(csv_path: str = "pokemon_tcg_battle_dataset.csv", model_out: str = "action_value_model.joblib"):
    df = load_dataset(csv_path)
    df = build_training_frame(df)

    action_dummies = pd.get_dummies(df["action_type"], prefix="action_type_raw")
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = pd.concat([df[feature_cols], action_dummies], axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R^2: {r2:.4f}")

    top_features = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])[:10]
    print("Top feature importances:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")

    joblib.dump({"model": model, "columns": list(X.columns)}, model_out)
    print(f"Saved model to {model_out}")
    return model


if __name__ == "__main__":
    train()

