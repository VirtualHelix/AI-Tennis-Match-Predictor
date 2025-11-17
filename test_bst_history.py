import sys
sys.path.append("src")

from trees.player_history import PlayerMatchHistory

# Fake simple rows (like pandas Series)
match1 = {"winner_name": "Nadal", "loser_name": "Djokovic", "surface": "Clay"}
match2 = {"winner_name": "Djokovic", "loser_name": "Nadal", "surface": "Hard"}
match3 = {"winner_name": "Djokovic", "loser_name": "Federer", "surface": "Grass"}

history = PlayerMatchHistory("Djokovic")

history.add_match("2013-06-06", match1)
history.add_match("2015-01-29", match2)
history.add_match("2014-07-12", match3)

print("All matches sorted by date:")
print(history.get_all_matches_sorted())

print("\nLast 2 matches:")
print(history.get_last_n_matches(2))

print("\nDjokovic on Hard:")
print(history.get_surface_record("Hard"))