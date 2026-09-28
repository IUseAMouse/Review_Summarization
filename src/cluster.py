import re
from collections import Counter

import numpy as np
from fastembed import TextEmbedding
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering

from config import (
    DISTANCE_THRESHOLD,
    EMBEDDING_MODEL,
    MAX_CLUSTERS,
    MAX_THEMES,
    MIN_CLUSTER_SIZE,
    QUOTES_PER_THEME,
    SYNTHESIS_MODEL,
)
from llm import ask
from prompts import THEMES_SYSTEM, THEMES_USER


class Theme(BaseModel):
    title: str
    summary: str
    cluster_ids: list[int]


class Themes(BaseModel):
    themes: list[Theme]


def embed(texts):
    model = TextEmbedding(EMBEDDING_MODEL)
    return np.array(list(model.embed(texts)))


def make_clusters(vectors):
    """Returns lists of insight indices, largest cluster first."""
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD,
    ).fit_predict(vectors)

    clusters = {}
    for index, label in enumerate(labels):
        clusters.setdefault(label, []).append(index)
    clusters = [c for c in clusters.values() if len(c) >= MIN_CLUSTER_SIZE]
    return sorted(clusters, key=len, reverse=True)[:MAX_CLUSTERS]


def most_central(indices, vectors):
    centroid = vectors[indices].mean(axis=0)
    return sorted(indices, key=lambda i: -(vectors[i] @ centroid))


def describe(clusters, insights, vectors):
    blocks = []
    for n, cluster in enumerate(clusters):
        sample = most_central(cluster, vectors)[:8]
        lines = "\n".join(f"- {insights[i]['insight']}" for i in sample)
        blocks.append(f'<group id="{n}" size="{len(cluster)}">\n{lines}\n</group>')
    return "\n\n".join(blocks)


def pick_quotes(indices, insights, vectors):
    # one quote per review, and no quote twice (copy-pasted reviews)
    quotes, seen = [], set()
    for i in most_central(indices, vectors):
        keys = {insights[i]["review_id"], re.sub(r"\W+", " ", insights[i]["quote"].lower()).strip()}
        if not keys & seen:
            seen |= keys
            quotes.append(insights[i])
        if len(quotes) == QUOTES_PER_THEME:
            break
    return quotes


def build_themes(client, insights, n_reviews):
    """Returns the themes ranked by number of reviews. Tags each insight with its theme."""
    if len(insights) < 2:
        raise SystemExit("Not enough insights found in the reviews to build a report.")

    print("Grouping insights...")
    vectors = embed([i["insight"] for i in insights])
    clusters = make_clusters(vectors)
    if not clusters:
        raise SystemExit("No recurring topic found. Try a larger set of reviews.")

    prompt = THEMES_USER.format(clusters=describe(clusters, insights, vectors))
    result = ask(client, SYNTHESIS_MODEL, THEMES_SYSTEM, prompt, Themes)

    themes, used = [], set()
    for theme in result.themes:
        ids = [c for c in dict.fromkeys(theme.cluster_ids) if 0 <= c < len(clusters) and c not in used]
        used.update(ids)
        indices = [i for c in ids for i in clusters[c]]
        if not indices:
            continue
        for i in indices:
            insights[i]["theme"] = theme.title
        reviews = {insights[i]["review_id"] for i in indices}
        themes.append(
            {
                "title": theme.title,
                "summary": theme.summary,
                "reviews": len(reviews),
                "share": len(reviews) / n_reviews,
                "sentiment": Counter(insights[i]["sentiment"] for i in indices),
                "quotes": pick_quotes(indices, insights, vectors),
            }
        )

    if not themes:
        raise SystemExit("No theme could be built from the reviews.")
    themes.sort(key=lambda t: t["reviews"], reverse=True)
    return themes[:MAX_THEMES]
