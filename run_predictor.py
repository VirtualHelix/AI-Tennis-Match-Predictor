import sys
sys.path.append("src")

from data_loader import load_all_matches
from preprocess import clean_matches
from trees.player_history_manager import PlayerHistoryManager
from elo import EloSystem
from features import build_feature_row
from model.predict import load_model, predict_from_features

from rapidfuzz import process


# ---------------------------------------------------------
# 1. FUZZY PLAYER NAME MATCHING
# ---------------------------------------------------------
def normalize_name(input_name, player_list, score_cutoff=60):
    input_name = input_name.strip()
    match, score, _ = process.extractOne(input_name, player_list)
    return match if score >= score_cutoff else input_name


# ---------------------------------------------------------
# 2. HEAD-TO-HEAD RECORD
# ---------------------------------------------------------
def head_to_head(df, p1, p2):
    h2h = df[
        ((df['winner_name'] == p1) & (df['loser_name'] == p2)) |
        ((df['winner_name'] == p2) & (df['loser_name'] == p1))
    ]
    wins_p1 = (h2h['winner_name'] == p1).sum()
    wins_p2 = (h2h['winner_name'] == p2).sum()
    return wins_p1, wins_p2


# ---------------------------------------------------------
# 3. LAST 5 MATCHES (W/L)
# ---------------------------------------------------------
def last_n_results(df, player, n=5):
    matches = df[
        (df['winner_name'] == player) | (df['loser_name'] == player)
    ].sort_values("tourney_date", ascending=False)

    results = []
    for _, row in matches.head(n).iterrows():
        results.append("W" if row["winner_name"] == player else "L")
    return results


# ---------------------------------------------------------
# 4. BUILD STATE (history + Elo)
# ---------------------------------------------------------
def build_state_until(df):
    manager = PlayerHistoryManager()
    elo = EloSystem()

    for _, row in df.iterrows():
        manager.add_match_from_row(row)
        elo.update_from_match(row)

    return manager, elo


# ---------------------------------------------------------
# 5. MAKE A PREDICTION BETWEEN TWO PLAYERS
# ---------------------------------------------------------
def predict_future_match(p1, p2, surface, df):
    manager, elo = build_state_until(df)

    histA = manager._get_player(p1)
    histB = manager._get_player(p2)

    fake_row = {"winner_name": p1, "loser_name": p2, "surface": surface}
    features = build_feature_row(fake_row, histA, histB, elo)

    model = load_model()
    return predict_from_features(features, model)


# ---------------------------------------------------------
# 6. MAIN USER INTERFACE
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading ATP dataset...")
    df = load_all_matches("data/raw")
    df = clean_matches(df)

    # Build list for name matching
    player_list = sorted(set(df["winner_name"]) | set(df["loser_name"]))

    # User input
    p1 = input("Player A: ")
    p2 = input("Player B: ")
    surface = input("Surface (Hard/Clay/Grass): ")

    # Fuzzy match names
    p1 = normalize_name(p1, player_list)
    p2 = normalize_name(p2, player_list)

    print(f"\nInterpreting players as: {p1} vs {p2}")

    # Head-to-Head
    winsA, winsB = head_to_head(df, p1, p2)
    print(f"Head-to-Head: {p1} {winsA} – {winsB} {p2}")

    # Last 5 matches
    print(f"Last 5 – {p1}: {' '.join(last_n_results(df, p1))}")
    print(f"Last 5 – {p2}: {' '.join(last_n_results(df, p2))}")

    # Prediction
    prob = predict_future_match(p1, p2, surface, df)

    if prob >= 0.5:
        winner = p1
        pct = prob * 100
    else:
        winner = p2
        pct = (1 - prob) * 100

    print(f"\nPredicted Winner: {winner} ({pct:.2f}%)")