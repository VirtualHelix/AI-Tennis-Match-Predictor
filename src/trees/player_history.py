from trees.bst import BinarySearchTree

class PlayerMatchHistory:
    """
    Stores all matches for a single player using a BST
    where the key = match date (sortable)
    and the value = match row (pandas Series)
    """

    def __init__(self, player_name):
        self.player_name = player_name
        self.tree = BinarySearchTree()

    def add_match(self, match_date, match_row):
        """
        Inserts a match into the player's BST.
        match_date should be a comparable value (string or datetime).
        match_row is a pandas Series of match info.
        """
        self.tree.insert(match_date, match_row)

    def get_all_matches_sorted(self):
        """Returns all matches sorted by date."""
        return self.tree.inorder()

    def get_last_n_matches(self, n):
        """Return last n matches (most recent first)."""
        all_matches = self.tree.inorder()  # oldest → newest
        return list(reversed(all_matches[-n:]))  # newest → oldest

    def get_surface_record(self, surface):
        """
        Returns (wins, losses) for the player on a given surface.
        """
        wins = 0
        losses = 0

        for date, match in self.tree.inorder():
            if match['surface'] == surface:
                if match['winner_name'] == self.player_name:
                    wins += 1
                elif match['loser_name'] == self.player_name:
                    losses += 1

        return wins, losses