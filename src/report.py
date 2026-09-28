from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from config import RESULTS_DIR, SYNTHESIS_MODEL
from llm import ask
from prompts import SUMMARY_SYSTEM, SUMMARY_USER

SENTIMENTS = ["positive", "negative", "mixed"]


class Summary(BaseModel):
    summary: str = Field(description="The full text of the executive summary")


def sentiment_split(theme):
    total = sum(theme["sentiment"].values())
    return [f"{theme['sentiment'][s] / total:.0%}" for s in SENTIMENTS]


def executive_summary(client, themes, n_reviews):
    lines = []
    for t in themes:
        pos, neg, mixed = sentiment_split(t)
        lines.append(
            f"- {t['title']}: {t['reviews']} reviews ({t['share']:.0%}), "
            f"{pos} positive, {neg} negative, {mixed} mixed. {t['summary']}"
        )
    prompt = SUMMARY_USER.format(reviews=n_reviews, themes="\n".join(lines))
    return ask(client, SYNTHESIS_MODEL, SUMMARY_SYSTEM, prompt, Summary).summary


def render(source, n_reviews, insights, rejected, themes, summary):
    in_theme = sum("theme" in i for i in insights)
    out = [
        "# Steam review analysis",
        "",
        f"Source: `{source.name}`, generated on {datetime.now():%Y-%m-%d at %H:%M}.",
        "",
        "| Reviews analysed | Insights extracted | Insights in a theme | Insights rejected |",
        "|---|---|---|---|",
        f"| {n_reviews} | {len(insights)} | {in_theme} | {rejected} |",
        "",
        "An insight is rejected when its quote cannot be found word for word in the review. "
        "Insights on topics too rare to form a theme are listed in the CSV file without a theme.",
        "",
        "## Executive summary",
        "",
        summary,
        "",
        "## Themes ranked by importance",
        "",
        "Share is the part of analysed reviews that mention the theme. "
        "A review can mention several themes.",
        "",
        "| # | Theme | Reviews | Share | Positive | Negative | Mixed |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, t in enumerate(themes, 1):
        pos, neg, mixed = sentiment_split(t)
        out.append(
            f"| {n} | {t['title']} | {t['reviews']} | {t['share']:.0%} | {pos} | {neg} | {mixed} |"
        )

    for n, t in enumerate(themes, 1):
        out += ["", f"## {n}. {t['title']}", "", t["summary"], ""]
        out += [f"Mentioned in {t['reviews']} reviews ({t['share']:.0%})."]
        for q in t["quotes"]:
            quote = " ".join(q["quote"].split())
            out += ["", f'> "{quote}"', f"> — row {q['review_id']}, {q['sentiment']}"]

    return "\n".join(out) + "\n"


def write_report(client, source, n_reviews, insights, rejected, themes):
    summary = executive_summary(client, themes, n_reviews)
    stamp = f"{datetime.now():%Y-%m-%d_%H%M}"

    report = RESULTS_DIR / f"report_{stamp}.md"
    RESULTS_DIR.mkdir(exist_ok=True)
    report.write_text(
        render(source, n_reviews, insights, rejected, themes, summary), encoding="utf-8"
    )

    table = pd.DataFrame(insights).rename(columns={"review_id": "row"})
    table = table.reindex(columns=["theme", "row", "sentiment", "insight", "quote"])
    # utf-8-sig so that Excel reads accents correctly
    table.to_csv(RESULTS_DIR / f"insights_{stamp}.csv", index=False, encoding="utf-8-sig")
    return report
