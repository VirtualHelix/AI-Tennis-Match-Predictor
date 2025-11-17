from trees.player_history import PlayerMatchHistory

class PlayerHistoryManager:
    """
    Builds and stores match histories for all players.
    """

    def __init__(self):
        self.players = {}

    def _get_player(self, name):
        """Return existing PlayerMatchHistory or create a new one."""
        if name not in self.players:
            self.players[name] = PlayerMatchHistory(name)
        return self.players[name]

    def add_match_from_row(self, row):
        """
        Given a pandas row representing a match,
        insert it into both players' histories.
        """
        date = row['tourney_date'] if 'tourney_date' in row else row['tourney_id']

        winner = self._get_player(row['winner_name'])
        loser = self._get_player(row['loser_name'])

        # Insert match into each player's history tree
        winner.add_match(date, row)
        loser.add_match(date, row)

    def build_histories(self, df):
        """
        Build match histories for all players from a dataframe.
        """
        for _, row in df.iterrows():
            self.add_match_from_row(row)