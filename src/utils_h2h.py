def head_to_head(df, playerA, playerB):
    """
    Returns winsA, winsB for all past matches between the two players.
    """
    h2h_df = df[
        ((df['winner_name'] == playerA) & (df['loser_name'] == playerB)) |
        ((df['winner_name'] == playerB) & (df['loser_name'] == playerA))
    ]

    winsA = ((h2h_df['winner_name'] == playerA)).sum()
    winsB = ((h2h_df['winner_name'] == playerB)).sum()

    return winsA, winsB