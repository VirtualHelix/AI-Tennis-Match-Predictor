import sys
sys.path.append("src")

from data_loader import load_all_matches
from preprocess import clean_matches
from trees.player_history_manager import PlayerHistoryManager

from data_loader import load_all_matches
from preprocess import clean_matches
from trees.player_history_manager import PlayerHistoryManager

df = load_all_matches("data/raw")
df = clean_matches(df)

manager = PlayerHistoryManager()
manager.build_histories(df.iloc[:500])  # first 500 matches for testing

print("Number of players:", len(manager.players))
print("Example players:", list(manager.players.keys())[:10])

djoko = manager.players.get("Novak Djokovic")
if djoko:
    print("Djokovic last 5:", djoko.get_last_n_matches(5))