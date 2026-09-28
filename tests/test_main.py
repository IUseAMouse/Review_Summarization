import anthropic
import pandas as pd
import pytest

import main
from test_llm import api_error

TOPICS = {"ai": 40, "multiplayer": 25, "mods": 15, "price": 1}


@pytest.fixture
def pipeline(fake_llm, fake_embed, monkeypatch, write_csv):
    monkeypatch.setattr(main, "get_client", lambda: None)
    rows, n = [], 0
    for topic, count in TOPICS.items():
        for _ in range(count):
            rows.append(f'{n},"{topic} is bad, review number {n}"')
            n += 1
    write_csv("ID,REVIEW TRANSLATED\n" + "\n".join(rows) + "\n")
    return fake_llm


def test_full_run(pipeline, project_dirs, capsys):
    main.main()

    out = capsys.readouterr().out
    assert "81 reviews loaded from reviews.csv" in out
    assert "81 insights extracted, 0 rejected" in out

    assert pipeline.count("Extraction") == 5
    assert pipeline.count("Themes") == 1
    assert pipeline.count("Summary") == 1

    reports = list((project_dirs / "results").glob("report_*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "| 81 | 81 | 80 | 0 |" in text
    assert "| 1 | Theme 0 | 40 | 49% | 0% | 100% | 0% |" in text
    assert "| 2 | Theme 1 | 25 | 31% |" in text
    assert "| 3 | Theme 2 | 15 | 19% |" in text
    assert "Theme 3" not in text
    assert text.count("> — row ") == 15

    table = pd.read_csv(next((project_dirs / "results").glob("insights_*.csv")))
    assert len(table) == 81
    assert table["theme"].value_counts().to_dict() == {"Theme 0": 40, "Theme 1": 25, "Theme 2": 15}


def test_quotes_in_report_come_from_the_cited_row(pipeline, project_dirs):
    main.main()
    _, reviews = main.load_reviews()
    table = pd.read_csv(next((project_dirs / "results").glob("insights_*.csv")))
    for row, quote in zip(table["row"], table["quote"]):
        assert quote in reviews[row]


def test_second_run_does_not_extract_again(pipeline):
    main.main()
    pipeline.calls.clear()
    main.main()
    assert pipeline.count("Extraction") == 0
    assert pipeline.count("Themes") == 1


def test_run_without_csv(fake_llm, monkeypatch):
    monkeypatch.setattr(main, "get_client", lambda: None)
    with pytest.raises(SystemExit, match="No .csv file"):
        main.main()


@pytest.mark.parametrize(
    "error, message",
    [
        (api_error(anthropic.AuthenticationError), "Delete the .env file"),
        (api_error(anthropic.PermissionDeniedError), "not allowed"),
        (api_error(anthropic.RateLimitError), "Rate limit"),
        (api_error(anthropic.APIConnectionError), "internet connection"),
        (api_error(anthropic.APITimeoutError), "internet connection"),
        (RuntimeError("Incomplete answer from model (max_tokens)"), "Incomplete answer"),
        (KeyboardInterrupt(), "Stopped"),
    ],
)
def test_errors_become_readable_messages(monkeypatch, error, message):
    def fail():
        raise error

    monkeypatch.setattr(main, "run", fail)
    with pytest.raises(SystemExit, match=message):
        main.main()


def test_other_api_errors_show_status(monkeypatch):
    error = api_error(anthropic.InternalServerError)
    error.status_code = 500
    error.message = "Overloaded"

    def fail():
        raise error

    monkeypatch.setattr(main, "run", fail)
    with pytest.raises(SystemExit, match=r"\(500\): Overloaded"):
        main.main()
