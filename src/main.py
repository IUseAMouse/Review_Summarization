import sys

import anthropic

from cluster import build_themes
from data import load_reviews
from extract import extract_insights
from llm import get_client
from report import write_report


def run():
    client = get_client()
    source, reviews = load_reviews()
    print(f"{len(reviews)} reviews loaded from {source.name}")

    insights, rejected = extract_insights(client, reviews)
    print(f"{len(insights)} insights extracted, {rejected} rejected")

    themes = build_themes(client, insights, len(reviews))
    report = write_report(client, source, len(reviews), insights, rejected, themes)
    print(f"Done. Report saved in {report}")


def main():
    try:
        run()
    except anthropic.AuthenticationError:
        sys.exit("The API key was rejected. Delete the .env file and run again.")
    except anthropic.PermissionDeniedError:
        sys.exit("This API key is not allowed to use the models set in src/config.py.")
    except anthropic.RateLimitError:
        sys.exit("Rate limit reached. Wait a few minutes and run again, progress is saved.")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Anthropic API. Check your internet connection.")
    except anthropic.APIStatusError as e:
        sys.exit(f"The Anthropic API returned an error ({e.status_code}): {e.message}")
    except RuntimeError as e:
        sys.exit(f"{e}. Run again to resume, progress is saved.")
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run again to resume, progress is saved.")


if __name__ == "__main__":
    main()
