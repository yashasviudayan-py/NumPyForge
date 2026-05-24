"""Deterministic local data ingestion for Phase 5 workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config_value, load_json_config  # noqa: E402
from src.types import FloatArray, IntArray  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.json"


def create_demo_dataset(
    *,
    n_samples_per_class: int = 40,
    random_state: int = 42,
) -> tuple[FloatArray, IntArray]:
    """Create a deterministic binary classification dataset."""
    rng = np.random.default_rng(random_state)
    negative = rng.normal(loc=(-1.0, -0.8), scale=0.35, size=(n_samples_per_class, 2))
    positive = rng.normal(loc=(1.0, 0.9), scale=0.35, size=(n_samples_per_class, 2))
    features = cast(FloatArray, np.vstack((negative, positive)).astype(np.float64))
    targets = cast(
        IntArray,
        np.concatenate(
            (
                np.zeros(n_samples_per_class, dtype=np.int_),
                np.ones(n_samples_per_class, dtype=np.int_),
            )
        ),
    )
    permutation = rng.permutation(features.shape[0])
    return features[permutation], targets[permutation]


def ingest(config_path: Path = DEFAULT_CONFIG) -> Path:
    """Create the configured processed dataset and return its path."""
    config = load_json_config(config_path)
    output_path = PROJECT_ROOT / get_config_value(config, "data.processed_path", str)
    random_state = get_config_value(config, "data.random_state", int)
    features, targets = create_demo_dataset(random_state=random_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X=features,
        y=targets,
        feature_names=np.array(["x1", "x2"]),
        target_name=np.array("label"),
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    """Run local ingestion and print a compact JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    output_path = ingest(args.config)
    with np.load(output_path, allow_pickle=False) as dataset:
        summary = {
            "processed_path": str(output_path),
            "n_samples": int(dataset["X"].shape[0]),
            "n_features": int(dataset["X"].shape[1]),
            "class_counts": {
                str(label): int(count)
                for label, count in zip(*np.unique(dataset["y"], return_counts=True), strict=True)
            },
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
