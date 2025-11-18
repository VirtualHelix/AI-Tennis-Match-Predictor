def recent_win_rate(player_history, n=10):
    """
    Computes recent win rate over last n matches.
    Returns a value from 0.0 to 1.0.
    """
    matches = player_history.get_last_n_matches(n)
    if len(matches) == 0:
        return 0.5  # neutral if no history

    wins = 0
    for date, match in matches:
        if match['winner_name'] == player_history.player_name:
            wins += 1

    return wins / len(matches)


def surface_win_rate(player_history, surface):
    """
    Computes win % on a given surface.
    """
    wins, losses = player_history.get_surface_record(surface)
    total = wins + losses
    if total == 0:
        return 0.5
    return wins / total


def head_to_head(playerA, playerB, matches_for_A):
    """
    Computes head-to-head win difference (A wins - B wins)
    using A's match history.
    """
    a_wins = 0
    b_wins = 0

    for _, match in matches_for_A:
        if match['winner_name'] == playerA and match['loser_name'] == playerB:
            a_wins += 1
        elif match['winner_name'] == playerB and match['loser_name'] == playerA:
            b_wins += 1

    return a_wins - b_wins


def elo_feature_vector(elo_system, winner_name, loser_name, surface):
    """
    Returns [global_elo_diff, surface_elo_diff]
    """
    w_global = elo_system.get_rating(winner_name)
    l_global = elo_system.get_rating(loser_name)

    w_surface = elo_system.get_surface_rating(winner_name, surface)
    l_surface = elo_system.get_surface_rating(loser_name, surface)

    return [
        w_global - l_global,
        w_surface - l_surface,
    ]


def build_feature_row(row, winner_history, loser_history, elo_system):
    """
    Build a feature vector for a single match row using:
    - recent form diff
    - surface win% diff
    - head-to-head diff
    - global Elo diff
    - surface Elo diff
    """
    surface = row['surface']
    w_name = winner_history.player_name
    l_name = loser_history.player_name

    # Recent form
    w_recent = recent_win_rate(winner_history)
    l_recent = recent_win_rate(loser_history)

    # Surface form
    w_surface = surface_win_rate(winner_history, surface)
    l_surface = surface_win_rate(loser_history, surface)

    # Head-to-head (from winner's perspective)
    h2h = head_to_head(w_name, l_name, winner_history.get_all_matches_sorted())

    # Elo features
    elo_vec = elo_feature_vector(elo_system, w_name, l_name, surface)

    return [
        w_recent - l_recent,   # recent form diff
        w_surface - l_surface, # surface form diff
        h2h,                   # head-to-head diff
        *elo_vec               # global Elo diff, surface Elo diff
    ]
