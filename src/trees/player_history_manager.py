from .player_history import PlayerMatchHistory

class PlayerHistoryManager:
    def __init__(self):
        self.players = {}  # name → PlayerMatchHistory

    def _get_player(self, name):
        """Return existing PlayerMatchHistory or create a new one."""
        if name not in self.players:
            self.players[name] = PlayerMatchHistory(name)
        return self.players[name]

    def add_match_from_row(self, row):
        """Insert match into both players' histories."""
        winner = row['winner_name']
        loser = row['loser_name']
        date = row['tourney_date'] if 'tourney_date' in row else row['tourney_id']

        match_info = {
            "winner_name": winner,
            "loser_name": loser,
            "surface": row['surface']
        }

        # Add match to histories
        self._get_player(winner).add_match(date, match_info)
        self._get_player(loser).add_match(date, match_info)

    def build_histories(self, df):
        """Load entire dataframe into player histories."""
        for _, row in df.iterrows():
            self.add_match_from_row(row)
            