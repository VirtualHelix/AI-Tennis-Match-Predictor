import sys
import numpy as np

sys.path.append("src")

from data_loader import load_all_matches
from preprocess import clean_matches
from trees.player_history_manager import PlayerHistoryManager
from elo import EloSystem
from features import build_feature_row


def build_dataset(df, min_matches=10):
    """
    Build feature matrix X and label vector y.
    For each match, generate:
      - positive sample (winner vs loser)  → y = 1
      - negative sample (loser vs winner) → y = 0
    Only use matches where both players have enough history.
    """

    # Ensure chronological order
    if "tourney_date" in df.columns:
        df = df.sort_values("tourney_date").reset_index(drop=True)
    else:
        df = df.sort_values("tourney_id").reset_index(drop=True)

    manager = PlayerHistoryManager()
    elo = EloSystem()

    X = []
    y = []

    for idx, row in df.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]

        # Histories before match
        w_hist = manager._get_player(winner)
        l_hist = manager._get_player(loser)

        # Must have enough matches already played
        if (len(w_hist.get_all_matches_sorted()) >= min_matches and
            len(l_hist.get_all_matches_sorted()) >= min_matches):

            # --------------------------
            # POSITIVE SAMPLE (winner perspective)
            # --------------------------
            features_pos = build_feature_row(row, w_hist, l_hist, elo)
            X.append(features_pos)
            y.append(1)

            # --------------------------
            # NEGATIVE SAMPLE (loser perspective)
            # --------------------------
            features_neg = build_feature_row(row, l_hist, w_hist, elo)
            X.append(features_neg)
            y.append(0)

        # Update histories + Elo AFTER feature creation
        manager.add_match_from_row(row)
        elo.update_from_match(row)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    return X, y


if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_all_matches("data/raw")
    df = clean_matches(df)

    print("Building dataset...")
    X, y = build_dataset(df, min_matches=10)

    print("Feature matrix shape:", X.shape)
    print("Label vector shape:", y.shape)

    np.savez("data/processed/dataset_basic_features.npz", X=X, y=y)
    print("Saved dataset to data/processed/dataset_basic_features.npz")