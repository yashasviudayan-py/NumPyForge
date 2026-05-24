"""Evaluate a saved Phase 5 binary logistic model artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingest import create_demo_dataset  # noqa: E402
from src.artifacts import load_logistic_artifact  # noqa: E402
from src.config import Config, get_config_value, load_json_config  # noqa: E402
from src.metrics import classification_report_dict, log_loss  # noqa: E402
from src.model_selection import train_test_split  # noqa: E402
from src.types import FloatArray, IntArray  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evaluation.json"


def evaluate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Evaluate the configured model artifact against the deterministic holdout split."""
    config = load_json_config(config_path)
    processed_path = _ensure_dataset(config)
    features, targets = _load_dataset(processed_path)
    _, X_test, _, y_test = train_test_split(
        features,
        targets,
        test_size=get_config_value(config, "data.test_size", float),
        stratify=targets,
        random_state=get_config_value(config, "data.random_state", int),
    )

    artifact = load_logistic_artifact(
        PROJECT_ROOT / get_config_value(config, "artifact.artifact_dir", str)
    )
    probabilities = artifact.model.predict_proba(X_test)
    predictions = artifact.model.predict(X_test)
    report = classification_report_dict(y_test, predictions, probabilities[:, 1])
    report["log_loss"] = log_loss(y_test, probabilities)

    return {
        "artifact_dir": str(artifact.path),
        "model_version": artifact.metadata.get("version"),
        "n_test_samples": int(X_test.shape[0]),
        "classification": report,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run evaluation and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.config), indent=2, sort_keys=True))


def _ensure_dataset(config: Config) -> Path:
    processed_path = PROJECT_ROOT / get_config_value(config, "data.processed_path", str)
    if not processed_path.exists():
        features, targets = create_demo_dataset(
            random_state=get_config_value(config, "data.random_state", int)
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            processed_path,
            X=features,
            y=targets,
            feature_names=np.array(["x1", "x2"]),
            target_name=np.array("label"),
        )
    return processed_path


def _load_dataset(path: Path) -> tuple[FloatArray, IntArray]:
    with np.load(path, allow_pickle=False) as dataset:
        return (
            cast(FloatArray, dataset["X"].astype(np.float64)),
            cast(IntArray, dataset["y"].astype(np.int_)),
        )


if __name__ == "__main__":
    main()
