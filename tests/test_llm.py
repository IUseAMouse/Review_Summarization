from types import SimpleNamespace

import anthropic
import pytest
from pydantic import BaseModel

import llm


class Answer(BaseModel):
    text: str


def api_error(kind):
    # built without __init__, which needs a full HTTP response
    return kind.__new__(kind)


class FakeAnthropic:
    """Replaces anthropic.Anthropic. Only the key "good" is accepted."""

    created = []

    def __init__(self, api_key, max_retries):
        self.api_key = api_key
        self.max_retries = max_retries
        self.models = SimpleNamespace(list=self.list_models)
        FakeAnthropic.created.append(self)

    def list_models(self, limit):
        if self.api_key != "good":
            raise api_error(anthropic.AuthenticationError)


@pytest.fixture
def fake_anthropic(monkeypatch):
    FakeAnthropic.created = []
    monkeypatch.setattr(llm.anthropic, "Anthropic", FakeAnthropic)


def typed_keys(monkeypatch, keys):
    answers = iter(keys)
    monkeypatch.setattr(llm.getpass, "getpass", lambda prompt: next(answers))


def test_get_client_asks_and_saves_key(fake_anthropic, monkeypatch):
    typed_keys(monkeypatch, ["  good \n"])
    client = llm.get_client()
    assert client.api_key == "good"
    assert client.max_retries == 5
    assert llm.ENV_FILE.read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=good\n"


def test_get_client_asks_again_after_bad_key(fake_anthropic, monkeypatch, capsys):
    typed_keys(monkeypatch, ["bad", "", "good"])
    assert llm.get_client().api_key == "good"
    assert capsys.readouterr().out.count("rejected") == 2
    assert llm.ENV_FILE.read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=good\n"


def test_get_client_does_not_save_bad_key(fake_anthropic, monkeypatch):
    typed_keys(monkeypatch, ["bad"])
    with pytest.raises(StopIteration):
        llm.get_client()
    assert not llm.ENV_FILE.exists()


def test_get_client_reads_saved_key(fake_anthropic, monkeypatch):
    typed_keys(monkeypatch, ["good"])
    llm.get_client()
    typed_keys(monkeypatch, [])
    assert llm.get_client().api_key == "good"


def test_get_client_reads_env_file(fake_anthropic, monkeypatch):
    llm.ENV_FILE.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    typed_keys(monkeypatch, [])
    assert llm.get_client().api_key == "from-file"


def test_get_client_reads_environment(fake_anthropic, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    typed_keys(monkeypatch, [])
    assert llm.get_client().api_key == "from-env"
    assert not llm.ENV_FILE.exists()


def test_get_client_asks_when_env_file_has_no_key(fake_anthropic, monkeypatch):
    llm.ENV_FILE.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    typed_keys(monkeypatch, ["good"])
    assert llm.get_client().api_key == "good"


def test_get_client_lets_network_errors_through(fake_anthropic, monkeypatch):
    def offline(self, limit):
        raise api_error(anthropic.APIConnectionError)

    monkeypatch.setattr(FakeAnthropic, "list_models", offline)
    typed_keys(monkeypatch, ["good"])
    with pytest.raises(anthropic.APIConnectionError):
        llm.get_client()
    assert not llm.ENV_FILE.exists()


def fake_client(stop_reason, parsed_output, calls=None):
    def parse(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed_output)

    return SimpleNamespace(messages=SimpleNamespace(parse=parse))


def test_ask_returns_parsed_output():
    calls = []
    client = fake_client("end_turn", Answer(text="hello"), calls)
    assert llm.ask(client, "some-model", "system text", "user text", Answer) == Answer(text="hello")
    assert calls[0]["model"] == "some-model"
    assert calls[0]["system"] == "system text"
    assert calls[0]["messages"] == [{"role": "user", "content": "user text"}]
    assert calls[0]["output_format"] is Answer


@pytest.mark.parametrize(
    "stop_reason, parsed",
    [("max_tokens", None), ("max_tokens", Answer(text="cut")), ("refusal", None), ("end_turn", None)],
)
def test_ask_raises_on_incomplete_answer(stop_reason, parsed):
    with pytest.raises(RuntimeError, match=stop_reason):
        llm.ask(fake_client(stop_reason, parsed), "some-model", "s", "p", Answer)


def test_ask_accepts_real_sdk_arguments():
    """Fails if the installed SDK no longer accepts the arguments used by ask."""
    import inspect

    parameters = inspect.signature(anthropic.Anthropic(api_key="x").messages.parse).parameters
    assert {"model", "max_tokens", "system", "messages", "output_format"} <= set(parameters)
