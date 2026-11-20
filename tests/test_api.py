import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="FastAPI optional dependency not installed")
def test_api_has_health_and_assess_routes():
    from eb1a.api import app
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/assess" in paths
