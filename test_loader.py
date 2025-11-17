import sys
sys.path.append("src")

from data_loader import load_all_matches
from preprocess import clean_matches

# Load the raw data
df = load_all_matches("data/raw")

# Clean it
df = clean_matches(df)

# Print the first 5 rows and the shape
print(df.head())
print(df.shape)