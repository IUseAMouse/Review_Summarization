import pandas as pd

from config import DATA_DIR, REVIEW_COLUMN


def find_csv():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No .csv file found in {DATA_DIR}")
    if len(files) == 1:
        return files[0]

    for n, file in enumerate(files, 1):
        print(f"  {n}. {file.name}")
    choice = input("Several files found, which one? ")
    if not choice.isdigit() or not 1 <= int(choice) <= len(files):
        raise SystemExit("Invalid choice.")
    return files[int(choice) - 1]


def find_delimiter(header):
    counts = {sep: header.count(sep) for sep in [",", ";", "\t"]}
    sep = max(counts, key=counts.get)
    # single column: use a separator that never appears in text
    return sep if counts[sep] else "\x1f"


def read_csv(path):
    for encoding in ["utf-8-sig", "cp1252"]:
        try:
            with path.open(encoding=encoding) as f:
                sep = find_delimiter(f.readline())
            return pd.read_csv(path, sep=sep, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise SystemExit(f"Could not read {path.name}: {e}")
    raise SystemExit(f"Could not read {path.name}: unknown text encoding.")


def pick_column(df):
    for col in df.columns:
        if str(col).strip().upper() == REVIEW_COLUMN:
            return col

    lengths = {c: df[c].dropna().astype(str).str.len().mean() for c in df.columns}
    lengths = {c: n for c, n in lengths.items() if pd.notna(n)}
    if not lengths:
        raise SystemExit("The CSV file is empty.")
    col = max(lengths, key=lengths.get)
    print(f'Column "{REVIEW_COLUMN}" not found, using "{col}" instead.')
    return col


def load_reviews():
    """Returns the file path and the reviews as {spreadsheet row: text}."""
    path = find_csv()
    df = read_csv(path)
    texts = df[pick_column(df)].dropna().astype(str).str.strip()
    texts = texts[texts != ""].drop_duplicates()
    if texts.empty:
        raise SystemExit("No review text found in the CSV file.")
    # +2: header is row 1, first review is row 2
    return path, {i + 2: text for i, text in texts.items()}
