from unittest.mock import patch

import httpx
import pytest

from myfitness.xunji.common import XunjiHttpClient, XunjiNetworkError


class _FailingClient:
    def __init__(self, exc: Exception, **kwargs):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        raise self.exc


def test_winerror_10013_becomes_actionable_network_error():
    request = httpx.Request("POST", "https://api.xunjiapp.cn/open/body/query_gzip")
    transport_error = httpx.ConnectError("connect failed", request=request)
    transport_error.__cause__ = PermissionError(
        "[WinError 10013] 以一种访问权限不允许的方式做了一个访问套接字的尝试"
    )
    client = XunjiHttpClient(max_retries=1)

    with patch(
        "myfitness.xunji.common.httpx.Client",
        side_effect=lambda **kwargs: _FailingClient(transport_error, **kwargs),
    ), pytest.raises(XunjiNetworkError) as captured:
        client.post("https://api.xunjiapp.cn/test", "test-key", {})

    message = str(captured.value)
    assert "WinError 10013" in message
    assert "普通 PowerShell" in message
    assert "test-key" not in message


def test_timeout_becomes_xunji_network_error():
    request = httpx.Request("POST", "https://api.xunjiapp.cn/test")
    timeout_error = httpx.ConnectTimeout("timed out", request=request)
    client = XunjiHttpClient(max_retries=1)

    with patch(
        "myfitness.xunji.common.httpx.Client",
        side_effect=lambda **kwargs: _FailingClient(timeout_error, **kwargs),
    ), pytest.raises(XunjiNetworkError, match="超时"):
        client.post("https://api.xunjiapp.cn/test", "test-key", {})
