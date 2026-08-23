import json

import pytest

from myfitness.config import Settings
from myfitness.llm.factory import LlmUnavailableError, _parse_stream_line, stream_chat_completion


def test_parse_stream_line_delta():
    line = 'data: {"choices":[{"delta":{"content":"你"}}]}'
    assert _parse_stream_line(line) == "你"


def test_parse_stream_line_done():
    assert _parse_stream_line("data: [DONE]") is None


def test_stream_chat_completion(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "data: [DONE]",
    ]

    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield from lines

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, headers, json):
            assert json["stream"] is True
            return FakeStreamResponse()

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)

    tokens = list(
        stream_chat_completion(
            [{"role": "user", "content": "hi"}],
            settings=Settings(llm_api_key="sk-test", llm_model="test-model"),
        )
    )
    assert tokens == ["Hello"]


def test_stream_retries_on_503_then_succeeds(monkeypatch):
    attempts = {"n": 0}
    ok_lines = [
        'data: {"choices":[{"delta":{"content":"OK"}}]}',
        "data: [DONE]",
    ]

    class OkResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield from ok_lines

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FailResponse:
        def raise_for_status(self):
            import httpx

            request = httpx.Request("POST", "https://x/chat/completions")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("503", request=request, response=response)

        def iter_lines(self):
            return iter([])
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return FailResponse()
            return OkResponse()

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)
    monkeypatch.setattr("myfitness.llm.factory._time.sleep", lambda *_: None)

    tokens = list(
        stream_chat_completion(
            [{"role": "user", "content": "hi"}],
            settings=Settings(llm_api_key="sk-test", llm_model="test-model"),
        )
    )
    assert tokens == ["OK"]
    assert attempts["n"] == 2


def test_stream_raises_unavailable_after_retries_exhausted(monkeypatch):
    import httpx

    def make_fail_response():
        class FailResponse:
            def raise_for_status(self):
                request = httpx.Request("POST", "https://x/chat/completions")
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError("503", request=request, response=response)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return FailResponse()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return make_fail_response()

    monkeypatch.setattr("myfitness.llm.factory.httpx.Client", FakeClient)
    monkeypatch.setattr("myfitness.llm.factory._time.sleep", lambda *_: None)

    with pytest.raises(LlmUnavailableError):
        list(
            stream_chat_completion(
                [{"role": "user", "content": "hi"}],
                settings=Settings(llm_api_key="sk-test", llm_model="test-model"),
            )
        )
