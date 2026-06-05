"""Tests for the end-to-end portfolio demo script."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Protocol, cast


class PortfolioDemoModule(Protocol):
    def main(self) -> None: ...


def _load_portfolio_demo() -> PortfolioDemoModule:
    module_path = Path(__file__).resolve().parents[1] / "examples" / "portfolio_demo.py"
    spec = importlib.util.spec_from_file_location("portfolio_demo_under_test", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load portfolio demo from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(PortfolioDemoModule, module)


portfolio_demo = _load_portfolio_demo()


def test_portfolio_demo_uses_its_isolated_serving_config(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """The demo should not reuse api.main's module-level app if it was imported earlier."""
    monkeypatch.setenv("NUMPYFORGE_ARTIFACT_DIR", str(tmp_path / "missing-default-artifact"))

    import api.main

    importlib.reload(api.main)

    portfolio_demo.main()

    output = capsys.readouterr().out
    json_start = output.rfind('{\n  "api"')
    assert json_start != -1
    summary = json.loads(output[json_start:])

    assert summary["model_version"] == "portfolio-demo"
    assert summary["api"]["health_status"] == 200
    assert summary["api"]["ready_status"] == 200
    assert summary["api"]["metadata_status"] == 200
    assert summary["api"]["predict_status"] == 200
