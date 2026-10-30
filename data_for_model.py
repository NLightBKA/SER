import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "mock_posts.json"

#Split data into training set and testing set in the future, and might want to work on normalization before throw the data into train.py
def get_data_for_model() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    posts = get_data_for_model()
    print(f"Loaded {len(posts)} posts successfully.")
