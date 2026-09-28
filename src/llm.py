import getpass
import os

import anthropic
from dotenv import dotenv_values

from config import ENV_FILE


def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY") or dotenv_values(ENV_FILE).get("ANTHROPIC_API_KEY")
    if key:
        return anthropic.Anthropic(api_key=key, max_retries=5)

    while True:
        key = getpass.getpass("Anthropic API key (input is hidden): ").strip()
        client = anthropic.Anthropic(api_key=key, max_retries=5)
        try:
            client.models.list(limit=1)
        except anthropic.AuthenticationError:
            print("This key was rejected, try again.")
            continue
        ENV_FILE.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
        print("Key saved in .env, you will not be asked again.")
        return client


def ask(client, model, system, prompt, schema):
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"Incomplete answer from {model} ({response.stop_reason})")
    return response.parsed_output
