import pandas as pd

def clean_matches(df):
    """
    Cleans Sackmann match data by removing walkovers/retirements
    and fixing player names.
    """

    # Strip spaces from player names
    df['winner_name'] = df['winner_name'].str.strip()
    df['loser_name'] = df['loser_name'].str.strip()

    # Remove walkovers/retirements
    if 'comment' in df.columns:
        df = df[~df['comment'].str.contains("Walkover|Retired", na=False)]

    # Remove rows missing essential values
    df = df.dropna(subset=['winner_name', 'loser_name', 'surface'])

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    return df