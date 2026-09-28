import pandas as pd
import pytest

import data


def test_find_csv_without_file():
    with pytest.raises(SystemExit, match="No .csv file"):
        data.find_csv()


def test_find_csv_ignores_other_files(project_dirs, write_csv):
    (project_dirs / "data" / ".gitkeep").touch()
    (project_dirs / "data" / "notes.txt").write_text("hello")
    path = write_csv("REVIEW TRANSLATED\nNice\n")
    assert data.find_csv() == path


def test_find_csv_asks_when_several(write_csv, monkeypatch):
    write_csv("a\n1\n", name="a.csv")
    second = write_csv("a\n1\n", name="b.csv")
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert data.find_csv() == second


@pytest.mark.parametrize("answer", ["", "0", "3", "abc", "-1", "1.5"])
def test_find_csv_rejects_bad_choice(write_csv, monkeypatch, answer):
    write_csv("a\n1\n", name="a.csv")
    write_csv("a\n1\n", name="b.csv")
    monkeypatch.setattr("builtins.input", lambda _: answer)
    with pytest.raises(SystemExit, match="Invalid choice"):
        data.find_csv()


@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_read_csv_detects_delimiter(write_csv, sep):
    path = write_csv(f"ID{sep}REVIEW TRANSLATED\n1{sep}Great game\n2{sep}Too hard\n")
    df = data.read_csv(path)
    assert list(df.columns) == ["ID", "REVIEW TRANSLATED"]
    assert list(df["REVIEW TRANSLATED"]) == ["Great game", "Too hard"]


def test_read_csv_with_excel_bom(write_csv):
    path = write_csv("REVIEW TRANSLATED,ID\nGreat game,1\n", encoding="utf-8-sig")
    assert list(data.read_csv(path).columns) == ["REVIEW TRANSLATED", "ID"]


def test_read_csv_with_windows_encoding(write_csv):
    path = write_csv("ID,REVIEW TRANSLATED\n1,Très bon jeu à 10€\n", encoding="cp1252")
    assert data.read_csv(path)["REVIEW TRANSLATED"][0] == "Très bon jeu à 10€"


@pytest.mark.parametrize(
    "header, sep",
    [
        ("ID,REVIEW TRANSLATED\n", ","),
        ("ID;REVIEW TRANSLATED;NOTE, EXTRA\n", ";"),
        ("ID\tREVIEW TRANSLATED\n", "\t"),
        ("REVIEW TRANSLATED\n", "\x1f"),
        ("", "\x1f"),
    ],
)
def test_find_delimiter(header, sep):
    assert data.find_delimiter(header) == sep


def test_read_csv_malformed_file(write_csv):
    path = write_csv('ID,REVIEW TRANSLATED\n1,Fine\n2,"never closed\n')
    with pytest.raises(SystemExit, match="Could not read reviews.csv"):
        data.read_csv(path)


def test_read_csv_empty_file(write_csv):
    path = write_csv("")
    with pytest.raises(SystemExit, match="Could not read reviews.csv"):
        data.read_csv(path)


def test_pick_column_exact():
    df = pd.DataFrame({"ID": [1], "REVIEW": ["original text, longer than the other"], "REVIEW TRANSLATED": ["short"]})
    assert data.pick_column(df) == "REVIEW TRANSLATED"


@pytest.mark.parametrize("name", ["review translated", " REVIEW TRANSLATED ", "Review Translated"])
def test_pick_column_tolerates_case_and_spaces(name):
    df = pd.DataFrame({"ID": [1], name: ["text"]})
    assert data.pick_column(df) == name


def test_pick_column_falls_back_on_longest_text(capsys):
    df = pd.DataFrame(
        {
            "id": [1001, 1002],
            "author": ["bob", "alice"],
            "text": ["A long review about the game", None],
            "empty": [None, None],
        }
    )
    assert data.pick_column(df) == "text"
    assert "not found" in capsys.readouterr().out


def test_pick_column_on_empty_file():
    with pytest.raises(SystemExit, match="empty"):
        data.pick_column(pd.DataFrame({"REVIEW": []}))


def test_load_reviews_uses_spreadsheet_rows(write_csv):
    path = write_csv("ID,REVIEW TRANSLATED\n1,First\n2,Second\n3,Third\n")
    source, reviews = data.load_reviews()
    assert source == path
    assert reviews == {2: "First", 3: "Second", 4: "Third"}


def test_load_reviews_drops_empty_and_duplicates(write_csv):
    write_csv('ID,REVIEW TRANSLATED\n1,Good\n2,\n3,"   "\n4,Good\n5,  Bad  \n')
    _, reviews = data.load_reviews()
    assert reviews == {2: "Good", 6: "Bad"}


def test_load_reviews_keeps_commas_quotes_and_line_breaks(write_csv):
    write_csv('ID,REVIEW TRANSLATED\n1,"Good, but ""hard"".\nSecond line; really."\n2,Fine\n')
    _, reviews = data.load_reviews()
    assert reviews == {2: 'Good, but "hard".\nSecond line; really.', 3: "Fine"}


def test_load_reviews_single_column_file(write_csv):
    write_csv(
        "REVIEW TRANSLATED\n"
        '"Great game, great mods, great music."\n'
        "One more turn; then another; then sunrise.\n"
        "Simple review\n"
    )
    _, reviews = data.load_reviews()
    assert reviews == {
        2: "Great game, great mods, great music.",
        3: "One more turn; then another; then sunrise.",
        4: "Simple review",
    }


def test_load_reviews_with_numeric_reviews(write_csv):
    write_csv("ID,REVIEW TRANSLATED\n1,10\n2,Great\n")
    _, reviews = data.load_reviews()
    assert reviews == {2: "10", 3: "Great"}


def test_load_reviews_without_text(write_csv):
    write_csv("ID,REVIEW TRANSLATED\n1,\n2,\n")
    with pytest.raises(SystemExit, match="No review text"):
        data.load_reviews()


def test_load_reviews_header_only(write_csv):
    write_csv("ID,REVIEW TRANSLATED\n")
    with pytest.raises(SystemExit):
        data.load_reviews()


def test_load_reviews_at_real_volume(write_csv):
    rows = "".join(f'{n},"Review number {n}, with a comma"\n' for n in range(1500))
    write_csv("ID,REVIEW TRANSLATED\n" + rows)
    _, reviews = data.load_reviews()
    assert len(reviews) == 1500
    assert reviews[1501] == "Review number 1499, with a comma"
