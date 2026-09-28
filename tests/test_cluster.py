import numpy as np
import pytest

import cluster
from cluster import Theme, Themes
from conftest import topic_vector


def make_insights(spec):
    """spec: list of (review_id, topic, sentiment)."""
    return [
        {"review_id": r, "insight": f"Opinion about {t}", "quote": f"quote {r}", "sentiment": s}
        for r, t, s in spec
    ]


def vectors_for(topics):
    return np.array([topic_vector(t) for t in topics])


def test_make_clusters_groups_by_topic():
    topics = ["ai"] * 5 + ["mods"] * 3 + ["price"] * 4
    clusters = cluster.make_clusters(vectors_for(topics))
    assert clusters == [[0, 1, 2, 3, 4], [8, 9, 10, 11], [5, 6, 7]]


def test_make_clusters_drops_small_groups():
    topics = ["ai"] * 3 + ["mods"] * 2 + ["price"]
    assert cluster.make_clusters(vectors_for(topics)) == [[0, 1, 2]]


def test_make_clusters_is_capped(monkeypatch):
    monkeypatch.setattr(cluster, "MAX_CLUSTERS", 2)
    topics = ["ai"] * 5 + ["mods"] * 3 + ["price"] * 4
    assert cluster.make_clusters(vectors_for(topics)) == [[0, 1, 2, 3, 4], [8, 9, 10, 11]]


def test_make_clusters_with_identical_vectors():
    assert cluster.make_clusters(vectors_for(["ai"] * 4)) == [[0, 1, 2, 3]]


def test_most_central_puts_outlier_last():
    vectors = np.array([[1.0, 0.0], [0.6, 0.8], [0.98, 0.199], [0.99, 0.141]])
    order = cluster.most_central([0, 1, 2, 3], vectors)
    assert sorted(order) == [0, 1, 2, 3]
    assert order[-1] == 1


def test_describe_lists_groups():
    insights = make_insights([(2, "ai", "negative"), (3, "ai", "negative"), (4, "mods", "positive")])
    vectors = vectors_for(["ai", "ai", "mods"])
    text = cluster.describe([[0, 1], [2]], insights, vectors)
    assert '<group id="0" size="2">' in text
    assert '<group id="1" size="1">' in text
    assert text.count("- Opinion about ai") == 2


def test_describe_limits_sample():
    insights = make_insights([(i, "ai", "negative") for i in range(30)])
    text = cluster.describe([list(range(30))], insights, vectors_for(["ai"] * 30))
    assert 'size="30"' in text
    assert text.count("- Opinion") == 8


def test_pick_quotes_one_per_review():
    insights = make_insights([(2, "ai", "negative")] * 3 + [(3, "ai", "negative")])
    quotes = cluster.pick_quotes([0, 1, 2, 3], insights, vectors_for(["ai"] * 4))
    assert sorted(q["review_id"] for q in quotes) == [2, 3]


def test_pick_quotes_skips_repeated_quotes():
    insights = make_insights([(i, "ai", "negative") for i in range(4)])
    for insight, quote in zip(insights, ["Same quote", "same  QUOTE.", "Other", "Same quote!"]):
        insight["quote"] = quote
    quotes = cluster.pick_quotes([0, 1, 2, 3], insights, vectors_for(["ai"] * 4))
    assert len(quotes) == 2
    assert {q["quote"].lower().split()[0] for q in quotes} == {"same", "other"}


def test_pick_quotes_is_capped():
    insights = make_insights([(i, "ai", "negative") for i in range(20)])
    quotes = cluster.pick_quotes(list(range(20)), insights, vectors_for(["ai"] * 20))
    assert len(quotes) == cluster.QUOTES_PER_THEME


def test_build_themes_ranks_by_reviews(fake_llm, fake_embed):
    spec = [(i, "mods", "positive") for i in range(3)] + [(i, "ai", "negative") for i in range(10, 16)]
    insights = make_insights(spec)
    themes = cluster.build_themes(None, insights, n_reviews=20)

    assert [t["reviews"] for t in themes] == [6, 3]
    assert themes[0]["share"] == pytest.approx(0.3)
    assert themes[0]["sentiment"] == {"negative": 6}
    assert themes[1]["sentiment"] == {"positive": 3}
    assert all(q["review_id"] >= 10 for q in themes[0]["quotes"])
    assert fake_llm.calls[0]["model"] == cluster.SYNTHESIS_MODEL


def test_build_themes_counts_reviews_not_insights(fake_llm, fake_embed):
    spec = [(2, "ai", "negative")] * 4 + [(3, "ai", "positive")]
    themes = cluster.build_themes(None, make_insights(spec), n_reviews=10)
    assert themes[0]["reviews"] == 2
    assert themes[0]["share"] == pytest.approx(0.2)
    assert themes[0]["sentiment"] == {"negative": 4, "positive": 1}


def test_build_themes_merges_groups(fake_llm, fake_embed):
    spec = [(i, "mods", "positive") for i in range(3)] + [(i, "ai", "negative") for i in range(10, 14)]
    fake_llm.themes = lambda prompt: Themes(
        themes=[Theme(title="Everything", summary="All.", cluster_ids=[0, 1])]
    )
    insights = make_insights(spec)
    themes = cluster.build_themes(None, insights, n_reviews=7)
    assert len(themes) == 1
    assert themes[0]["reviews"] == 7
    assert themes[0]["share"] == 1
    assert {i["theme"] for i in insights} == {"Everything"}


def test_build_themes_ignores_invalid_group_ids(fake_llm, fake_embed):
    spec = [(i, "mods", "positive") for i in range(3)] + [(i, "ai", "negative") for i in range(10, 14)]
    fake_llm.themes = lambda prompt: Themes(
        themes=[
            Theme(title="AI", summary="s", cluster_ids=[0, 0, 42, -1]),
            Theme(title="AI again", summary="s", cluster_ids=[0]),
            Theme(title="Nothing", summary="s", cluster_ids=[]),
            Theme(title="Mods", summary="s", cluster_ids=[1, 0]),
        ]
    )
    themes = cluster.build_themes(None, make_insights(spec), n_reviews=7)
    assert [(t["title"], t["reviews"]) for t in themes] == [("AI", 4), ("Mods", 3)]


def test_build_themes_tags_only_grouped_insights(fake_llm, fake_embed):
    spec = [(i, "ai", "negative") for i in range(4)] + [(9, "price", "mixed")]
    insights = make_insights(spec)
    cluster.build_themes(None, insights, n_reviews=5)
    assert [i.get("theme") for i in insights] == ["Theme 0"] * 4 + [None]


def test_build_themes_is_capped(fake_llm, fake_embed, monkeypatch):
    monkeypatch.setattr(cluster, "MAX_THEMES", 1)
    spec = [(i, "mods", "positive") for i in range(3)] + [(i, "ai", "negative") for i in range(10, 14)]
    themes = cluster.build_themes(None, make_insights(spec), n_reviews=7)
    assert [t["reviews"] for t in themes] == [4]


@pytest.mark.parametrize("n", [0, 1])
def test_build_themes_needs_insights(fake_llm, fake_embed, n):
    with pytest.raises(SystemExit, match="Not enough insights"):
        cluster.build_themes(None, make_insights([(2, "ai", "negative")] * n), n_reviews=5)
    assert fake_llm.calls == []


def test_build_themes_without_recurring_topic(fake_llm, fake_embed):
    spec = [(2, "ai", "negative"), (3, "mods", "positive"), (4, "price", "mixed")]
    with pytest.raises(SystemExit, match="No recurring topic"):
        cluster.build_themes(None, make_insights(spec), n_reviews=3)
    assert fake_llm.calls == []


def test_build_themes_when_model_returns_nothing(fake_llm, fake_embed):
    fake_llm.themes = lambda prompt: Themes(themes=[])
    with pytest.raises(SystemExit, match="No theme"):
        cluster.build_themes(None, make_insights([(i, "ai", "negative") for i in range(4)]), n_reviews=4)


@pytest.mark.slow
def test_real_embeddings_separate_topics():
    texts = [
        "The AI makes poor military decisions.",
        "Computer opponents are bad at war.",
        "The AI never uses its army properly.",
        "Multiplayer games keep disconnecting.",
        "Online matches crash and desync all the time.",
        "Multiplayer is unstable and drops players.",
    ]
    vectors = cluster.embed(texts)
    assert vectors.shape[0] == 6
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)

    similarity = vectors @ vectors.T
    assert similarity[0, 1] > similarity[0, 3]
    assert similarity[3, 4] > similarity[3, 0]

    clusters = cluster.make_clusters(vectors)
    assert all(set(c) <= {0, 1, 2} or set(c) <= {3, 4, 5} for c in clusters)
