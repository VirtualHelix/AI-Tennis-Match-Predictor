import os
import pandas as pd

def load_all_matches(raw_folder_path):
    """
    Reads every 'atp_matches_YEAR.csv' file from Jeff Sackmann's dataset
    and combines them into one DataFrame.
    """

    files = os.listdir(raw_folder_path)

    match_files = sorted([
        f for f in files
        if f.startswith("atp_matches_") and f.endswith(".csv")
    ])

    dfs = []

    for file in match_files:
        full_path = os.path.join(raw_folder_path, file)
        print(f"Loading {file}...")

        df = pd.read_csv(full_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    print("Loaded", len(match_files), "files.")
    print("Total rows:", combined_df.shape[0])

    return combined_df