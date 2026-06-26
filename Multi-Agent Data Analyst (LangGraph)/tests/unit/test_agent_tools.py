"""Smoke tests for the three HTTP wrappers in ``agents.agent_tools``."""
import pytest
import requests

from agents import agent_tools
from agents.agent_tools import reset_short_memory, run_code, run_sql


def _response(payload, status_code=200):
    resp = type("Resp", (), {})()
    resp.status_code = status_code
    resp.json = lambda p=payload: p
    resp.text = str(payload)
    return resp


class TestRunCode:
    def test_posts_to_python_endpoint(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return _response({"status": "success", "output": "ok"})

        monkeypatch.setattr(agent_tools.requests, "post", fake_post)
        run_code("print('hi')", "out_var", False)

        assert captured["url"] == "http://localhost:8000/run_python"
        assert captured["json"] == {
            "code": "print('hi')",
            "output_name": "out_var",
            "last_step": False,
        }


class TestRunSql:
    def test_posts_to_sql_endpoint(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return _response({"status": "success", "output": "ok"})

        monkeypatch.setattr(agent_tools.requests, "post", fake_post)
        run_sql("SELECT 1", "rows", True)

        assert captured["url"] == "http://localhost:8000/run_sql"
        assert captured["json"] == {
            "code": "SELECT 1",
            "output_name": "rows",
            "last_step": True,
        }


class TestResetShortMemory:
    def test_posts_to_reset_endpoint_without_payload(self, monkeypatch):
        captured = {}

        def fake_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return _response({"status": "ok"})

        monkeypatch.setattr(agent_tools.requests, "post", fake_post)
        reset_short_memory()

        assert captured["url"] == "http://localhost:8000/reset"
        assert captured["json"] is None
