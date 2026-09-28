from collections import Counter
from pathlib import Path

import pandas as pd

import report


def make_theme(title="Late-game AI", reviews=30, share=0.3, sentiment=None, quotes=None):
    return {
        "title": title,
        "summary": f"Summary of {title}.",
        "reviews": reviews,
        "share": share,
        "sentiment": Counter(sentiment or {"negative": 3, "positive": 1}),
        "quotes": quotes
        or [{"review_id": 12, "quote": "the AI is dumb", "sentiment": "negative", "insight": "x"}],
    }


def make_insights():
    return [
        {"review_id": 12, "insight": "AI is weak", "quote": "the AI is dumb", "sentiment": "negative", "theme": "Late-game AI"},
        {"review_id": 15, "insight": "Pricey DLC", "quote": "too expensive", "sentiment": "negative"},
    ]


def test_sentiment_split():
    assert report.sentiment_split(make_theme()) == ["25%", "75%", "0%"]


def test_sentiment_split_single_sentiment():
    assert report.sentiment_split(make_theme(sentiment={"mixed": 2})) == ["0%", "0%", "100%"]


def test_executive_summary_sends_figures(fake_llm):
    themes = [make_theme(), make_theme(title="Mods", reviews=10, share=0.1)]
    assert report.executive_summary(None, themes, 100) == "Players like the game."
    prompt = fake_llm.calls[0]["prompt"]
    assert "100 reviews analysed" in prompt
    assert "- Late-game AI: 30 reviews (30%), 25% positive, 75% negative, 0% mixed." in prompt
    assert "- Mods: 10 reviews (10%)" in prompt
    assert fake_llm.calls[0]["model"] == report.SYNTHESIS_MODEL


def test_render_content():
    themes = [make_theme(), make_theme(title="Mods", reviews=10, share=0.1)]
    text = report.render(Path("civ5.csv"), 100, make_insights(), 4, themes, "The summary.")

    assert text.startswith("# Steam review analysis\n")
    assert "Source: `civ5.csv`" in text
    assert "| 100 | 2 | 1 | 4 |" in text
    assert "## Executive summary\n\nThe summary.\n" in text
    assert "| 1 | Late-game AI | 30 | 30% | 25% | 75% | 0% |" in text
    assert "| 2 | Mods | 10 | 10% | 25% | 75% | 0% |" in text
    assert "## 1. Late-game AI\n\nSummary of Late-game AI.\n" in text
    assert "## 2. Mods\n" in text
    assert "Mentioned in 30 reviews (30%)." in text
    assert '> "the AI is dumb"\n> — row 12, negative\n' in text
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "\n\n\n" not in text


def test_render_puts_quotes_on_one_line():
    quote = {"review_id": 5, "quote": "first line\nsecond   line", "sentiment": "mixed", "insight": "x"}
    text = report.render(Path("a.csv"), 1, [], 0, [make_theme(quotes=[quote])], "s")
    assert '> "first line second line"\n> — row 5, mixed' in text


def test_write_report_creates_files(fake_llm, project_dirs):
    path = report.write_report(None, Path("civ5.csv"), 100, make_insights(), 4, [make_theme()])

    assert path.parent == project_dirs / "results"
    assert path.name.startswith("report_") and path.suffix == ".md"
    assert "## 1. Late-game AI" in path.read_text(encoding="utf-8")
    assert "Players like the game." in path.read_text(encoding="utf-8")

    tables = list((project_dirs / "results").glob("insights_*.csv"))
    assert len(tables) == 1
    table = pd.read_csv(tables[0], encoding="utf-8-sig")
    assert list(table.columns) == ["theme", "row", "sentiment", "insight", "quote"]
    assert list(table["row"]) == [12, 15]
    assert table["theme"][0] == "Late-game AI"
    assert pd.isna(table["theme"][1])


def test_write_report_when_no_insight_has_a_theme(fake_llm, project_dirs):
    insights = [i for i in make_insights() if "theme" not in i]
    report.write_report(None, Path("civ5.csv"), 100, insights, 0, [make_theme()])
    table = pd.read_csv(next((project_dirs / "results").glob("insights_*.csv")))
    assert list(table.columns) == ["theme", "row", "sentiment", "insight", "quote"]


def test_write_report_keeps_non_ascii_text(fake_llm):
    quote = {"review_id": 5, "quote": "Très bon jeu 👍 — 10€", "sentiment": "positive", "insight": "x"}
    theme = make_theme(title="Prix élevé", quotes=[quote])
    path = report.write_report(None, Path("civ5.csv"), 1, [quote], 0, [theme])
    text = path.read_bytes().decode("utf-8")
    assert "## 1. Prix élevé" in text
    assert '> "Très bon jeu 👍 — 10€"' in text
