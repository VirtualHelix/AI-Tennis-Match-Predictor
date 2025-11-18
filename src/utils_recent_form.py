def last_n_results(df, player, n=5):
    """
    Returns a list of the player's last n match results as 'W' or 'L'.
    Sorted with newest first.
    """
    # Filter matches where player participated
    pl = df[
        (df["winner_name"] == player) |
        (df["loser_name"] == player)
    ].sort_values("tourney_date", ascending=False)

    results = []
    for _, row in pl.head(n).iterrows():
        if row["winner_name"] == player:
            results.append("W")
        else:
            results.append("L")

    return results