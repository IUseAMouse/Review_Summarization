import json

import pytest

import extract
from extract import Extraction, Insight


def make_reviews(n):
    return {i + 2: f"Review{i} says the ai is bad at war" for i in range(n)}


@pytest.mark.parametrize(
    "quote, review",
    [
        ("the AI is dumb", "I think the AI is dumb."),
        ("THE ai IS dumb", "I think the AI is dumb."),
        ("the AI   is\ndumb", "I think the AI is dumb."),
        ("the AI is dumb", "I think the AI\n\nis   dumb."),
        ("  the AI is dumb  ", "I think the AI is dumb."),
        ("c'est génial 👍", "Franchement c'est génial 👍 !"),
    ],
)
def test_is_verbatim_accepts(quote, review):
    assert extract.is_verbatim(quote, review)


@pytest.mark.parametrize(
    "quote, review",
    [
        ("the AI is stupid", "I think the AI is dumb."),
        ("the AI is dumb and slow", "I think the AI is dumb."),
        ("I think dumb", "I think the AI is dumb."),
        ("", "I think the AI is dumb."),
        ("   ", "I think the AI is dumb."),
        ("anything", ""),
    ],
)
def test_is_verbatim_rejects(quote, review):
    assert not extract.is_verbatim(quote, review)


def test_batch_key_depends_on_content():
    reviews = {2: "a", 3: "b"}
    assert extract.batch_key(reviews, [2, 3]) == extract.batch_key(dict(reviews), [2, 3])
    assert extract.batch_key(reviews, [2, 3]) != extract.batch_key(reviews, [2])
    assert extract.batch_key(reviews, [2, 3]) != extract.batch_key({2: "a", 3: "c"}, [2, 3])


def test_extract_batch_sends_every_review(fake_llm):
    reviews = {2: "First review text here", 7: "Second review\non two lines"}
    insights = extract.extract_batch(None, reviews, [2, 7])
    prompt = fake_llm.calls[0]["prompt"]
    assert '<review id="2">\nFirst review text here\n</review>' in prompt
    assert '<review id="7">\nSecond review\non two lines\n</review>' in prompt
    assert fake_llm.calls[0]["model"] == extract.EXTRACTION_MODEL
    assert [i["review_id"] for i in insights] == [2, 7]
    assert set(insights[0]) == {"review_id", "insight", "quote", "sentiment"}


@pytest.mark.parametrize("n, calls", [(1, 1), (20, 1), (21, 2), (45, 3)])
def test_extract_insights_batches(fake_llm, n, calls):
    insights, rejected = extract.extract_insights(None, make_reviews(n))
    assert len(fake_llm.calls) == calls
    assert len(insights) == n
    assert rejected == 0
    assert [i["review_id"] for i in insights] == sorted(i["review_id"] for i in insights)


def test_extract_insights_rejects_unsourced(fake_llm):
    reviews = {2: "The ai is bad at war", 3: "Mods are great"}
    fake_llm.extraction = lambda prompt: Extraction(
        insights=[
            Insight(review_id=2, insight="ok", quote="ai is bad", sentiment="negative"),
            Insight(review_id=2, insight="invented", quote="ai is terrible", sentiment="negative"),
            Insight(review_id=3, insight="wrong review", quote="ai is bad", sentiment="negative"),
            Insight(review_id=99, insight="unknown review", quote="Mods", sentiment="positive"),
            Insight(review_id=3, insight="empty quote", quote="", sentiment="positive"),
        ]
    )
    insights, rejected = extract.extract_insights(None, reviews)
    assert [i["insight"] for i in insights] == ["ok"]
    assert rejected == 4


def test_extract_insights_without_any_insight(fake_llm):
    fake_llm.extraction = lambda prompt: Extraction(insights=[])
    assert extract.extract_insights(None, make_reviews(3)) == ([], 0)


def test_extract_insights_uses_cache_on_second_run(fake_llm):
    reviews = make_reviews(45)
    first = extract.extract_insights(None, reviews)
    second = extract.extract_insights(None, reviews)
    assert len(fake_llm.calls) == 3
    assert first == second


def test_extract_insights_resumes_after_failure(fake_llm, monkeypatch):
    monkeypatch.setattr(extract, "MAX_WORKERS", 1)
    reviews = make_reviews(60)
    fake_llm.fail_on_call = 2
    with pytest.raises(RuntimeError):
        extract.extract_insights(None, reviews)
    done = len(extract.load_cache())
    assert 1 <= done < 3

    fake_llm.fail_on_call = None
    fake_llm.calls.clear()
    insights, _ = extract.extract_insights(None, reviews)
    assert len(fake_llm.calls) == 3 - done
    assert len(insights) == 60


def test_extract_insights_redoes_batch_when_review_changes(fake_llm):
    reviews = make_reviews(45)
    extract.extract_insights(None, reviews)
    reviews[2] = "A different review about mods"
    fake_llm.calls.clear()
    extract.extract_insights(None, reviews)
    assert len(fake_llm.calls) == 1


def test_load_cache_without_file():
    assert extract.load_cache() == {}


def test_load_cache_skips_damaged_lines():
    good = {"key": "abc", "insights": [{"review_id": 2}]}
    extract.CACHE_FILE.parent.mkdir()
    extract.CACHE_FILE.write_text(
        "\n".join([json.dumps(good), "", '{"key": "cut", "insi', '{"other": 1}', "[1, 2]", "null"]),
        encoding="utf-8",
    )
    assert extract.load_cache() == {"abc": [{"review_id": 2}]}


def test_cache_survives_damaged_file(fake_llm):
    reviews = make_reviews(5)
    extract.extract_insights(None, reviews)
    with extract.CACHE_FILE.open("a", encoding="utf-8") as f:
        f.write('{"key": "cut", "insi')
    fake_llm.calls.clear()
    insights, _ = extract.extract_insights(None, reviews)
    assert fake_llm.calls == []
    assert len(insights) == 5


def test_cache_keeps_non_ascii_text(fake_llm):
    reviews = {2: "Très bon jeu 👍 vraiment, je le conseille à tous"}
    first, _ = extract.extract_insights(None, reviews)
    second, _ = extract.extract_insights(None, reviews)
    assert first == second
    assert first[0]["quote"] == "Très bon jeu 👍 vraiment, je"
