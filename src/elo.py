import math

class EloSystem:
    def __init__(self, base_rating=1500, k_factor=32):
        # Global Elo
        self.ratings = {}

        # Surface-specific Elo
        self.surface_ratings = {
            "Hard": {},
            "Clay": {},
            "Grass": {}
        }

        self.base_rating = base_rating
        self.k_factor = k_factor

    # ------------------------------------
    # Surface Normalization
    # ------------------------------------
    def normalize_surface(self, surface):
        """
        Normalize surface labels from dataset.
        Treat 'Carpet' as 'Hard'.
        """
        s = str(surface).strip().title()

        if s == "Carpet":
            return "Hard"  # treat carpet as indoor hard

        if s in ["Hard", "H"]:
            return "Hard"
        if s in ["Clay", "C"]:
            return "Clay"
        if s in ["Grass", "G"]:
            return "Grass"

        # fallback
        return "Hard"

    # ------------------------------------
    # Basic Elo Helpers
    # ------------------------------------
    def get_rating(self, player):
        return self.ratings.get(player, self.base_rating)

    def set_rating(self, player, rating):
        self.ratings[player] = rating

    def expected_score(self, ratingA, ratingB):
        return 1 / (1 + 10 ** ((ratingB - ratingA) / 400))

    # ------------------------------------
    # Surface-Specific Elo Helpers
    # ------------------------------------
    def get_surface_rating(self, player, surface):
        s = self.normalize_surface(surface)
        return self.surface_ratings[s].get(player, self.base_rating)

    def set_surface_rating(self, player, surface, rating):
        s = self.normalize_surface(surface)
        self.surface_ratings[s][player] = rating

    # ------------------------------------
    # Elo Update From Match
    # ------------------------------------
    def update_from_match(self, row):
        winner = row['winner_name']
        loser = row['loser_name']
        surface = self.normalize_surface(row['surface'])

        # ----- GLOBAL ELO -----
        w_global = self.get_rating(winner)
        l_global = self.get_rating(loser)

        w_exp = self.expected_score(w_global, l_global)
        l_exp = 1 - w_exp

        w_new = w_global + self.k_factor * (1 - w_exp)
        l_new = l_global + self.k_factor * (0 - l_exp)

        self.set_rating(winner, w_new)
        self.set_rating(loser, l_new)

        # ----- SURFACE ELO -----
        w_surf = self.get_surface_rating(winner, surface)
        l_surf = self.get_surface_rating(loser, surface)

        w_exp_surf = self.expected_score(w_surf, l_surf)
        l_exp_surf = 1 - w_exp_surf

        w_new_surf = w_surf + self.k_factor * (1 - w_exp_surf)
        l_new_surf = l_surf + self.k_factor * (0 - l_exp_surf)

        self.set_surface_rating(winner, surface, w_new_surf)
        self.set_surface_rating(loser, surface, l_new_surf)

    # ------------------------------------
    # Surface Win Probability
    # ------------------------------------
    def surface_probability(self, playerA, playerB, surface):
        s = self.normalize_surface(surface)
        rA = self.get_surface_rating(playerA, s)
        rB = self.get_surface_rating(playerB, s)
        return self.expected_score(rA, rB)