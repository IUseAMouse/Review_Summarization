import re

import numpy as np
import pytest

import cluster
import data
import extract
import llm
import report

TOPICS = ["ai", "multiplayer", "mods", "price"]


@pytest.fixture(autouse=True)
def project_dirs(tmp_path, monkeypatch):
    """Points every path of the project to a temporary folder."""
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(extract, "CACHE_FILE", tmp_path / ".cache" / "extraction.jsonl")
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


def topic_vector(text):
    """One axis per topic, so that texts on the same topic land in the same cluster."""
    vector = np.full(len(TOPICS) + 1, 0.01)
    for axis, topic in enumerate(TOPICS):
        if topic in text.lower().split():
            vector[axis] = 1.0
    return vector / np.linalg.norm(vector)


@pytest.fixture
def fake_embed(monkeypatch):
    monkeypatch.setattr(cluster, "embed", lambda texts: np.array([topic_vector(t) for t in texts]))


class FakeLLM:
    """Stands for llm.ask in every module. Records the calls it receives."""

    def __init__(self):
        self.calls = []
        self.fail_on_call = None

    def __call__(self, client, model, system, prompt, schema):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema.__name__})
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("Incomplete answer")
        return getattr(self, schema.__name__.lower())(prompt)

    def count(self, schema):
        return sum(call["schema"] == schema for call in self.calls)

    def extraction(self, prompt):
        found = re.findall(r'<review id="(\d+)">\n(.*?)\n</review>', prompt, re.DOTALL)
        insights = [
            extract.Insight(
                review_id=int(review_id),
                insight=f"Players talk about {text.split()[0]} in the game",
                quote=" ".join(text.split()[:6]),
                sentiment="negative" if "bad" in text else "positive",
            )
            for review_id, text in found
        ]
        return extract.Extraction(insights=insights)

    def themes(self, prompt):
        ids = [int(n) for n in re.findall(r'<group id="(\d+)"', prompt)]
        themes = [
            cluster.Theme(title=f"Theme {n}", summary=f"Summary {n}.", cluster_ids=[n])
            for n in ids
        ]
        return cluster.Themes(themes=themes)

    def summary(self, prompt):
        return report.Summary(summary="Players like the game.")


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    for module in (extract, cluster, report):
        monkeypatch.setattr(module, "ask", fake)
    return fake


@pytest.fixture
def write_csv(project_dirs):
    def write(content, name="reviews.csv", encoding="utf-8"):
        path = project_dirs / "data" / name
        path.write_bytes(content.encode(encoding))
        return path

    return write
