# ReviewSummary

Reads a file of Steam reviews and writes a report of the main themes players
talk about, ranked by importance, each backed by quotes from the reviews.

## Setup (once)

1. Install `uv`, the tool that runs the script.

   Windows, in PowerShell:

   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   macOS, in Terminal:

   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Close the window and open a new one afterwards.

2. Download this project and open a terminal in its folder.

3. Install what the script needs:

   ```
   uv sync
   ```

## Run an analysis

1. Put your `.csv` file in the `data` folder. The reviews must be in a column
   named `REVIEW TRANSLATED`.

2. Run:

   ```
   uv run src/main.py
   ```

3. The first time, paste your Anthropic API key when asked and press Enter.
   Nothing shows on screen while you paste, this is normal. The key is saved in
   a file named `.env` on your computer and is not asked again.

With 1500 reviews, a run takes a few minutes. The first run also downloads a
small language model (about 70 MB).

## Results

Two files appear in the `results` folder:

| File | Content |
|---|---|
| `report_<date>.md` | The report: summary, ranked themes, quotes |
| `insights_<date>.csv` | Every point found in the reviews, with its theme and quote. Opens in Excel. |

`row` is the row of the review in your CSV file as shown in Excel, so you can
check any quote against the original.

## If something goes wrong

| Message | What to do |
|---|---|
| `No .csv file found` | Put your file in the `data` folder. |
| `The API key was rejected` | Delete the `.env` file, run again, paste a valid key. |
| `Rate limit reached` | Wait a few minutes and run again. |
| `Column "REVIEW TRANSLATED" not found, using ...` | Not an error. Check that the column named in the message holds the reviews. |

If a run stops halfway, run it again: reviews already read are not read twice.
To start from scratch, delete the `.cache` folder.

## Settings

Models, batch size and number of themes are in `src/config.py`.

## Tests

```
uv run --group test pytest
```
