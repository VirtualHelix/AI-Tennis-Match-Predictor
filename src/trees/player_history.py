class PlayerMatchHistory:
    def __init__(self, player_name):
        self.player_name = player_name
        self.matches = []  # list of (date, match_dict)

    def add_match(self, date, match_dict):
        """Store match as (date, match_dict)."""
        self.matches.append((date, match_dict))

    def get_all_matches_sorted(self):
        """Return all matches sorted oldest → newest."""
        return sorted(self.matches, key=lambda x: x[0])

    def get_last_n_matches(self, n):
        """Return the last n matches (most recent)."""
        matches = self.get_all_matches_sorted()
        return matches[-n:]

    def get_surface_record(self, surface):
        """Count wins/losses on a given surface."""
        wins = 0
        losses = 0

        for _, match in self.matches:
            if match['surface'] != surface:
                continue
            if match['winner_name'] == self.player_name:
                wins += 1
            elif match['loser_name'] == self.player_name:
                losses += 1

        return wins, losses