from __future__ import annotations

import pytest
from pydantic import BaseModel

from eda_agent.llm.client import LLMOutputError, OpenAICompatibleClient
from eda_agent.llm.config import LLMConfig


class Foo(BaseModel):
    bar: int


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Records every call and returns queued responses in order."""

    def __init__(self, contents: list[str | None]) -> None:
        self._contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._contents.pop(0)
        return _FakeResponse(content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAI:
    def __init__(self, contents: list[str | None]) -> None:
        self.completions = _FakeCompletions(contents)
        self.chat = _FakeChat(self.completions)


def _make_client(contents: list[str | None]) -> tuple[OpenAICompatibleClient, _FakeOpenAI]:
    fake = _FakeOpenAI(contents)
    config = LLMConfig(
        provider="ollama", base_url="http://fake", api_key="fake", model="fake-model"
    )
    client = OpenAICompatibleClient(config=config, client=fake)
    return client, fake


def test_complete_returns_parsed_instance_on_valid_first_response() -> None:
    client, fake = _make_client(['{"bar": 42}'])

    result = client.complete(system="sys", user="user", schema=Foo)

    assert isinstance(result, Foo)
    assert result.bar == 42
    assert len(fake.completions.calls) == 1


def test_complete_retries_once_and_succeeds_on_second_valid_response() -> None:
    client, fake = _make_client(['{"bar": "not an int and also missing brace"', '{"bar": 7}'])

    result = client.complete(system="sys", user="user", schema=Foo)

    assert isinstance(result, Foo)
    assert result.bar == 7
    assert len(fake.completions.calls) == 2

    retry_messages = fake.completions.calls[1]["messages"]
    retry_user_content = retry_messages[-1]["content"]
    assert "invalid" in retry_user_content.lower()
    # The validation/parse error from the first attempt must be included
    # so the model can correct itself.
    assert "bar" in retry_user_content


def test_complete_raises_llm_output_error_when_both_attempts_fail() -> None:
    client, fake = _make_client(["not json at all", "still not json"])

    with pytest.raises(LLMOutputError):
        client.complete(system="sys", user="user", schema=Foo)

    assert len(fake.completions.calls) == 2


def test_complete_raises_when_response_has_no_content() -> None:
    client, _fake = _make_client([None, None])

    with pytest.raises(LLMOutputError):
        client.complete(system="sys", user="user", schema=Foo)
