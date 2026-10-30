import math
from collections import Counter
from itertools import combinations


def get_tag_counts(data: list[dict]) -> dict[str, int]:
    counter = Counter()
    for post in data:
        counter.update(post.get("tags", []))
    return dict(counter.most_common())


def compute_idf(tag_counts: dict[str, int], n_posts: int) -> dict[str, float]:
    return {tag: math.log(n_posts / count) for tag, count in tag_counts.items()}


def pair_similarity(tags_a: list[str], tags_b: list[str], idf: dict[str, float]) -> float:
    shared = set(tags_a) & set(tags_b)
    return sum(idf.get(t, 0.0) for t in shared)


def normalize_scores(
    pairs_with_scores: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    scores = [s for _, _, s in pairs_with_scores]
    lo, hi = min(scores), max(scores)
    if hi > lo:
        return [(a, b, (s - lo) / (hi - lo)) for a, b, s in pairs_with_scores]
    return [(a, b, 0.0) for a, b, _ in pairs_with_scores]


def build_pairs(
    data: list[dict],
    tag_counts: dict[str, int] | None = None,
    normalize: bool = True,
) -> list[tuple[int, int, float]]:
    n_posts = len(data)
    if n_posts < 2:
        return []

    if tag_counts is None:
        tag_counts = get_tag_counts(data)

    idf = compute_idf(tag_counts, n_posts)

    raw_pairs = []
    for a, b in combinations(range(n_posts), 2):
        score = pair_similarity(data[a].get("tags", []), data[b].get("tags", []), idf)
        raw_pairs.append((a, b, score))

    if normalize:
        return normalize_scores(raw_pairs)
    return raw_pairs

