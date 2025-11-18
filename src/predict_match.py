import sys
sys.path.append("src")

from data_loader import load_all_matches
from preprocess import clean_matches
from trees.player_history_manager import PlayerHistoryManager
from elo import EloSystem
from features import build_feature_row
from model.predict import load_model, predict_from_features

# --- NEW IMPORTS ---
from rapidfuzz import process

def normalize_name(input_name, player_list, score_cutoff=60):
    """
    Fuzzy match an input name to the closest ATP player name.
    """
    input_name = input_name.strip()
    if not player_list:
        return input_name

    match, score, _ = process.extractOne(input_name, player_list)
    return match if score >= score_cutoff else input_name


def head_to_head(df, playerA, playerB):
    """
    Count H2H wins between playerA and playerB.
    """
    h2h_df = df[
        ((df['winner_name'] == playerA) & (df['loser_name'] == playerB)) |
        ((df['winner_name'] == playerB) & (df['loser_name'] == playerA))
    ]

    winsA = (h2h_df['winner_name'] == playerA).sum()
    winsB = (h2h_df['winner_name'] == playerB).sum()

    return winsA, winsB


def last_n_results(df, player, n=5):
    """
    Last n match results as W/L, newest first.
    """
    matches = df[
        (df['winner_name'] == player) |
        (df['loser_name'] == player)
    ].sort_values("tourney_date", ascending=False)

    results = []
    for _, row in matches.head(n).iterrows():
        results.append("W" if row['winner_name'] == player else "L")

    return results


def build_state_until(df):
    """
    Build full player history and Elo ratings from the dataset.
    """
    manager = PlayerHistoryManager()
    elo = EloSystem()

    for _, row in df.iterrows():
        manager.add_match_from_row(row)
        elo.update_from_match(row)

    return manager, elo


def predict_future_match(playerA, playerB, surface, df):
    """
    Predicts the probability that playerA beats playerB on a given surface.
    """
    manager, elo = build_state_until(df)

    A_hist = manager._get_player(playerA)
    B_hist = manager._get_player(playerB)

    fake_row = {
        "winner_name": playerA,
        "loser_name": playerB,
        "surface": surface
    }

    features = build_feature_row(fake_row, A_hist, B_hist, elo)
    model = load_model()

    return predict_from_features(features, model)


if __name__ == "__main__":
    df = load_all_matches("data/raw")
    df = clean_matches(df)

    # Grab list of all ATP players from dataset
    player_list = sorted(set(df["winner_name"].unique()) | set(df["loser_name"].unique()))

    p1 = input("Player A: ")
    p2 = input("Player B: ")
    surface = input("Surface (Hard/Clay/Grass): ")

    # --- FUZZY NAME MATCHING ---
    p1 = normalize_name(p1, player_list)
    p2 = normalize_name(p2, player_list)

    print(f"\nInterpreting players as: {p1} vs {p2}")

    # --- HEAD-TO-HEAD ---
    winsA, winsB = head_to_head(df, p1, p2)
    if winsA >= winsB:
        print(f"Head-to-Head: {p1} leads {winsA}-{winsB}")
    else:
        print(f"Head-to-Head: {p2} leads {winsB}-{winsA}")

    # --- LAST 5 MATCHES ---
    last5_A = last_n_results(df, p1, n=5)
    last5_B = last_n_results(df, p2, n=5)

    print(f"Last 5 Matches – {p1}: {' '.join(last5_A)}")
    print(f"Last 5 Matches – {p2}: {' '.join(last5_B)}")

    # --- PREDICTION ---
    prob = predict_future_match(p1, p2, surface, df)

    # Pick predicted winner + % formatting
    if prob >= 0.5:
        winner = p1
        pct = prob * 100
    else:
        winner = p2
        pct = (1 - prob) * 100

    print(f"\nPredicted Winner: {winner} ({pct:.2f}%)")