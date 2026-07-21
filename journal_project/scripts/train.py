"""
Generic Ultralytics YOLO experiment runner for the journal study.

All experiment settings are loaded from a YAML configuration file.

Examples
--------
Full baseline training:

    python journal_project/scripts/train.py \
        --config journal_project/configs/baseline.yaml

PowerShell single-line command:

    python journal_project/scripts/train.py --config journal_project/configs/baseline.yaml

Three-epoch smoke test:

    python journal_project/scripts/train.py \
        --config journal_project/configs/baseline.yaml \
        --smoke-test

PowerShell single-line command:

    python journal_project/scripts/train.py --config journal_project/configs/baseline.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train an Ultralytics YOLO model using a journal experiment "
            "configuration file."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to an experiment configuration YAML file, for example "
            "journal_project/configs/baseline.yaml."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Override the configured run with a three-epoch smoke test. "
            "The original configuration file is not modified."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration loading and validation
# ---------------------------------------------------------------------------


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and ensure it contains a dictionary."""

    if not path.exists():
        raise FileNotFoundError(f"YAML file was not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file but received: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            content = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid YAML syntax in {path}: {error}"
        ) from error

    if not isinstance(content, dict):
        raise ValueError(
            f"YAML file must contain a dictionary: {path}"
        )

    return content


def require_section(
    configuration: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Retrieve a required dictionary section."""

    section = configuration.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(
            f"Configuration section '{section_name}' is missing or "
            "is not a dictionary."
        )

    return section


def require_keys(
    section: dict[str, Any],
    section_name: str,
    required_keys: set[str],
) -> None:
    """Ensure a configuration section contains all required keys."""

    missing_keys = required_keys.difference(section)

    if missing_keys:
        raise ValueError(
            f"Configuration section '{section_name}' is missing keys: "
            f"{sorted(missing_keys)}"
        )


def validate_experiment_configuration(
    configuration: dict[str, Any],
) -> None:
    """Validate the complete experiment configuration."""

    experiment = require_section(
        configuration,
        "experiment",
    )

    model = require_section(
        configuration,
        "model",
    )

    dataset = require_section(
        configuration,
        "dataset",
    )

    training = require_section(
        configuration,
        "training",
    )

    reproducibility = require_section(
        configuration,
        "reproducibility",
    )

    runtime = require_section(
        configuration,
        "runtime",
    )

    output = require_section(
        configuration,
        "output",
    )

    require_keys(
        experiment,
        "experiment",
        {"group", "name", "description"},
    )

    require_keys(
        model,
        "model",
        {"weights"},
    )

    require_keys(
        dataset,
        "dataset",
        {"data"},
    )

    require_keys(
        training,
        "training",
        {
            "imgsz",
            "epochs",
            "batch",
            "optimizer",
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "patience",
        },
    )

    require_keys(
        reproducibility,
        "reproducibility",
        {"seed", "deterministic"},
    )

    require_keys(
        runtime,
        "runtime",
        {"device", "workers", "amp", "cache"},
    )

    require_keys(
        output,
        "output",
        {"plots", "save", "save_period"},
    )

    if not str(experiment["group"]).strip():
        raise ValueError(
            "experiment.group cannot be empty."
        )

    if not str(experiment["name"]).strip():
        raise ValueError(
            "experiment.name cannot be empty."
        )

    if not str(experiment["description"]).strip():
        raise ValueError(
            "experiment.description cannot be empty."
        )

    if not str(model["weights"]).strip():
        raise ValueError(
            "model.weights cannot be empty."
        )

    if not str(dataset["data"]).strip():
        raise ValueError(
            "dataset.data cannot be empty."
        )

    if int(training["epochs"]) <= 0:
        raise ValueError(
            "training.epochs must be greater than zero."
        )

    if int(training["batch"]) == 0:
        raise ValueError(
            "training.batch cannot be zero. Use a positive value "
            "or -1 for automatic batch sizing."
        )

    if int(training["imgsz"]) <= 0:
        raise ValueError(
            "training.imgsz must be greater than zero."
        )

    if int(training["patience"]) < 0:
        raise ValueError(
            "training.patience cannot be negative."
        )

    if float(training["lr0"]) <= 0:
        raise ValueError(
            "training.lr0 must be greater than zero."
        )

    if float(training["lrf"]) <= 0:
        raise ValueError(
            "training.lrf must be greater than zero."
        )

    if float(training["momentum"]) < 0:
        raise ValueError(
            "training.momentum cannot be negative."
        )

    if float(training["weight_decay"]) < 0:
        raise ValueError(
            "training.weight_decay cannot be negative."
        )

    if "workers" in training:
        raise ValueError(
            "The 'workers' setting belongs under the runtime section, "
            "not the training section."
        )

    if int(runtime["workers"]) < 0:
        raise ValueError(
            "runtime.workers cannot be negative."
        )

    if int(reproducibility["seed"]) < 0:
        raise ValueError(
            "reproducibility.seed cannot be negative."
        )

    save_period = int(output["save_period"])

    if save_period == 0 or save_period < -1:
        raise ValueError(
            "output.save_period must be -1 or a positive integer."
        )


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


def resolve_repository_root() -> Path:
    """
    Resolve the cloned Ultralytics repository root.

    Expected script location:
        <repository>/journal_project/scripts/train.py
    """

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]

    if not (
        repository_root / "pyproject.toml"
    ).exists():
        raise RuntimeError(
            "Could not identify the Ultralytics repository root. "
            f"Resolved candidate: {repository_root}"
        )

    return repository_root


def resolve_from_repository(
    value: str | Path,
    repository_root: Path,
) -> Path:
    """Resolve a relative path against the repository root."""

    path = Path(str(value)).expanduser()

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def resolve_model_source(
    model_value: str,
    repository_root: Path,
) -> str:
    """
    Resolve a local model path or retain an Ultralytics identifier.

    Examples retained as identifiers:
        yolo11s.pt
        yolo11s.yaml

    Examples resolved as local paths:
        journal_project/models/yolo11_ca.yaml
    """

    model_value = str(model_value).strip()

    if not model_value:
        raise ValueError(
            "model.weights cannot be empty."
        )

    candidate = Path(model_value).expanduser()

    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(
                "Configured model file was not found: "
                f"{candidate}"
            )

        return str(candidate.resolve())

    repository_candidate = (
        repository_root / candidate
    )

    if repository_candidate.exists():
        return str(repository_candidate.resolve())

    # Permit official identifiers such as yolo11s.pt.
    if candidate.parent == Path("."):
        return model_value

    raise FileNotFoundError(
        "Configured local model file was not found. "
        f"Checked: {repository_candidate.resolve()}"
    )


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Cannot hash a directory: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def get_optional_file_record(
    path: Path,
) -> dict[str, Any]:
    """Return path and hash information for an optional file."""

    if not path.exists():
        return {
            "available": False,
            "path": str(path),
            "sha256": None,
        }

    if not path.is_file():
        return {
            "available": False,
            "path": str(path),
            "sha256": None,
            "reason": "Path exists but is not a file.",
        }

    return {
        "available": True,
        "path": str(path.resolve()),
        "sha256": calculate_sha256(path),
        "size_bytes": path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def normalize_class_names(
    names: list[str] | dict[int | str, str],
) -> dict[int, str]:
    """Convert class names into an integer-keyed dictionary."""

    if isinstance(names, list):
        class_names = {
            class_id: str(class_name)
            for class_id, class_name in enumerate(names)
        }

    elif isinstance(names, dict):
        try:
            class_names = {
                int(class_id): str(class_name)
                for class_id, class_name in names.items()
            }

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Dataset class-name dictionary keys must be integers."
            ) from error

    else:
        raise ValueError(
            "The data.yaml 'names' field must be a list "
            "or dictionary."
        )

    if not class_names:
        raise ValueError(
            "The dataset contains no class names."
        )

    expected_ids = list(
        range(len(class_names))
    )

    if sorted(class_names) != expected_ids:
        raise ValueError(
            "Dataset class IDs must begin at zero and be contiguous. "
            f"Found: {sorted(class_names)}"
        )

    for class_id, class_name in class_names.items():
        if not class_name.strip():
            raise ValueError(
                "Dataset class name is empty for class ID "
                f"{class_id}."
            )

    return class_names


def resolve_dataset_entry(
    dataset_root: Path,
    value: str | list[str],
    entry_name: str,
) -> list[Path]:
    """
    Resolve one dataset split entry.

    Ultralytics dataset entries may be strings or lists of paths.
    """

    values = (
        value
        if isinstance(value, list)
        else [value]
    )

    if not values:
        raise ValueError(
            f"Dataset YAML entry '{entry_name}' cannot be empty."
        )

    resolved_paths: list[Path] = []

    for item in values:
        item_path = Path(
            str(item)
        ).expanduser()

        if not item_path.is_absolute():
            item_path = (
                dataset_root / item_path
            )

        resolved_paths.append(
            item_path.resolve()
        )

    return resolved_paths


def validate_dataset_paths(
    paths: list[Path],
    split_name: str,
) -> None:
    """Ensure every configured dataset split path exists."""

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{split_name.capitalize()} path was not found: "
                f"{path}"
            )

        if not path.is_dir():
            raise ValueError(
                f"{split_name.capitalize()} path is not a directory: "
                f"{path}"
            )


def locate_split_report(
    dataset_root: Path,
) -> Path:
    """Return the expected leakage-safe split-report path."""

    return (
        dataset_root
        / "split_manifests"
        / "split_report.json"
    )


def load_and_validate_split_report(
    split_report_path: Path,
) -> dict[str, Any]:
    """
    Load and validate the leakage-safe split report.

    The report is optional so the runner can still be used with
    other datasets.
    """

    if not split_report_path.exists():
        return {
            "available": False,
            "path": str(split_report_path),
            "sha256": None,
            "reason": "split_report.json was not found.",
        }

    if not split_report_path.is_file():
        raise ValueError(
            "Split report path is not a file: "
            f"{split_report_path}"
        )

    try:
        report = json.loads(
            split_report_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid JSON in split report "
            f"{split_report_path}: {error}"
        ) from error

    if not isinstance(report, dict):
        raise ValueError(
            "Split report must contain a JSON object: "
            f"{split_report_path}"
        )

    configuration = report.get(
        "configuration"
    )

    if (
        configuration is not None
        and not isinstance(configuration, dict)
    ):
        raise ValueError(
            "The split report 'configuration' field "
            "must be a dictionary."
        )

    split_information = report.get(
        "splits"
    )

    if split_information is not None:
        if not isinstance(
            split_information,
            dict,
        ):
            raise ValueError(
                "The split report 'splits' field "
                "must be a dictionary."
            )

        for required_split in (
            "train",
            "val",
            "test",
        ):
            if required_split not in split_information:
                raise ValueError(
                    "Split report is missing "
                    f"'{required_split}'."
                )

    return {
        "available": True,
        "path": str(
            split_report_path.resolve()
        ),
        "sha256": calculate_sha256(
            split_report_path
        ),
        "size_bytes": (
            split_report_path.stat().st_size
        ),
        "report": report,
    }


def load_and_validate_data_yaml(
    data_yaml: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[int, str],
    dict[str, list[Path]],
    dict[str, Any],
]:
    """
    Validate the dataset YAML and train, validation, and test paths.
    """

    content = load_yaml_file(
        data_yaml
    )

    required_keys = {
        "path",
        "train",
        "val",
        "test",
        "names",
    }

    missing_keys = (
        required_keys.difference(content)
    )

    if missing_keys:
        raise ValueError(
            "Dataset YAML is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    dataset_root = Path(
        str(content["path"])
    ).expanduser()

    if not dataset_root.is_absolute():
        dataset_root = (
            data_yaml.parent / dataset_root
        )

    dataset_root = dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root was not found: {dataset_root}"
        )

    if not dataset_root.is_dir():
        raise ValueError(
            f"Dataset root is not a directory: {dataset_root}"
        )

    split_paths = {
        "train": resolve_dataset_entry(
            dataset_root,
            content["train"],
            "train",
        ),
        "val": resolve_dataset_entry(
            dataset_root,
            content["val"],
            "val",
        ),
        "test": resolve_dataset_entry(
            dataset_root,
            content["test"],
            "test",
        ),
    }

    validate_dataset_paths(
        split_paths["train"],
        "training",
    )

    validate_dataset_paths(
        split_paths["val"],
        "validation",
    )

    validate_dataset_paths(
        split_paths["test"],
        "testing",
    )

    class_names = normalize_class_names(
        content["names"]
    )

    declared_nc = content.get("nc")

    if (
        declared_nc is not None
        and int(declared_nc) != len(class_names)
    ):
        raise ValueError(
            f"data.yaml declares nc={declared_nc}, but "
            f"{len(class_names)} class names were found."
        )

    split_report_path = locate_split_report(
        dataset_root
    )

    split_report_record = (
        load_and_validate_split_report(
            split_report_path
        )
    )

    return (
        content,
        dataset_root,
        class_names,
        split_paths,
        split_report_record,
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_random_seeds(
    seed: int,
) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_determinism(
    deterministic: bool,
) -> None:
    """
    Configure PyTorch deterministic behavior.

    Ultralytics also receives the deterministic flag.
    """

    if deterministic:
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )

        except TypeError:
            torch.use_deterministic_algorithms(
                True
            )

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    else:
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = False


def run_git_command(
    arguments: list[str],
    repository_root: Path,
) -> str:
    """Run a Git command and safely return its output."""

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        OSError,
    ):
        return "unavailable"


def get_gpu_information() -> list[dict[str, Any]]:
    """Return information about every visible CUDA device."""

    devices: list[dict[str, Any]] = []

    if not torch.cuda.is_available():
        return devices

    for device_index in range(
        torch.cuda.device_count()
    ):
        properties = (
            torch.cuda.get_device_properties(
                device_index
            )
        )

        devices.append(
            {
                "index": device_index,
                "name": properties.name,
                "total_memory_gb": round(
                    properties.total_memory
                    / 1024**3,
                    2,
                ),
                "compute_capability": (
                    f"{properties.major}."
                    f"{properties.minor}"
                ),
            }
        )

    return devices


def build_environment_record(
    repository_root: Path,
    config_path: Path,
    data_yaml: Path,
    dataset_root: Path,
    split_paths: dict[str, list[Path]],
    split_report_record: dict[str, Any],
    model_source: str,
    training_arguments: dict[str, Any],
) -> dict[str, Any]:
    """Build a reproducibility and environment record."""

    compact_split_report = {
        key: value
        for key, value in split_report_record.items()
        if key != "report"
    }

    cudnn_version = None

    if torch.backends.cudnn.is_available():
        cudnn_version = (
            torch.backends.cudnn.version()
        )

    return {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count()
        ),
        "cuda_devices": get_gpu_information(),
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "ultralytics_source": str(
            Path(
                ultralytics.__file__
            ).resolve()
        ),
        "git_commit": run_git_command(
            ["rev-parse", "HEAD"],
            repository_root,
        ),
        "git_branch": run_git_command(
            ["branch", "--show-current"],
            repository_root,
        ),
        "git_status": run_git_command(
            ["status", "--short"],
            repository_root,
        ),
        "git_remote_origin": run_git_command(
            ["remote", "get-url", "origin"],
            repository_root,
        ),
        "repository_root": str(
            repository_root
        ),
        "config_path": str(
            config_path
        ),
        "source_config_file": (
            get_optional_file_record(
                config_path
            )
        ),
        "data_yaml": str(
            data_yaml
        ),
        "data_yaml_file": (
            get_optional_file_record(
                data_yaml
            )
        ),
        "dataset_root": str(
            dataset_root
        ),
        "dataset_split_paths": {
            split_name: [
                str(path)
                for path in paths
            ]
            for split_name, paths
            in split_paths.items()
        },
        "dataset_split_report": (
            compact_split_report
        ),
        "model_source": model_source,
        "training_arguments": (
            training_arguments
        ),
    }


# ---------------------------------------------------------------------------
# Configuration conversion
# ---------------------------------------------------------------------------


def apply_smoke_test_overrides(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Create an in-memory three-epoch smoke-test configuration."""

    smoke_configuration = deepcopy(
        configuration
    )

    experiment = smoke_configuration[
        "experiment"
    ]

    training = smoke_configuration[
        "training"
    ]

    experiment["name"] = (
        f"{experiment['name']}_smoke"
    )

    experiment["description"] = (
        f"{experiment['description']} "
        "[three-epoch smoke test]"
    )

    training["epochs"] = 3
    training["patience"] = 3

    return smoke_configuration


def build_training_arguments(
    configuration: dict[str, Any],
    data_yaml: Path,
    project_directory: Path,
) -> dict[str, Any]:
    """Convert journal YAML settings to Ultralytics arguments."""

    experiment = configuration[
        "experiment"
    ]

    training = configuration[
        "training"
    ]

    reproducibility = configuration[
        "reproducibility"
    ]

    runtime = configuration[
        "runtime"
    ]

    output = configuration[
        "output"
    ]

    training_arguments: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": int(
            training["imgsz"]
        ),
        "epochs": int(
            training["epochs"]
        ),
        "batch": int(
            training["batch"]
        ),
        "optimizer": str(
            training["optimizer"]
        ),
        "lr0": float(
            training["lr0"]
        ),
        "lrf": float(
            training["lrf"]
        ),
        "momentum": float(
            training["momentum"]
        ),
        "weight_decay": float(
            training["weight_decay"]
        ),
        "patience": int(
            training["patience"]
        ),
        "seed": int(
            reproducibility["seed"]
        ),
        "deterministic": bool(
            reproducibility["deterministic"]
        ),
        "device": str(
            runtime["device"]
        ),
        "workers": int(
            runtime["workers"]
        ),
        "amp": bool(
            runtime["amp"]
        ),
        "cache": runtime["cache"],
        "project": str(
            project_directory
        ),
        "name": str(
            experiment["name"]
        ),
        "exist_ok": bool(
            output.get(
                "exist_ok",
                False,
            )
        ),
        "pretrained": bool(
            configuration["model"].get(
                "pretrained",
                True,
            )
        ),
        "plots": bool(
            output["plots"]
        ),
        "save": bool(
            output["save"]
        ),
        "save_period": int(
            output["save_period"]
        ),
        "verbose": bool(
            output.get(
                "verbose",
                True,
            )
        ),
    }

    optional_training_keys = {
        "cos_lr",
        "close_mosaic",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "box",
        "cls",
        "dfl",
        "nbs",
        "multi_scale",
        "rect",
        "single_cls",
        "fraction",
        "freeze",
        "resume",
        "val",
        "profile",
        "overlap_mask",
        "mask_ratio",
        "dropout",
        "iou",
        "max_det",
    }

    for key in optional_training_keys:
        if key in training:
            training_arguments[key] = (
                training[key]
            )

    return training_arguments


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------


def read_highest_map_epoch_summary(
    results_csv: Path,
) -> dict[str, Any]:
    """
    Identify the epoch with the highest validation mAP@0.5:0.95.

    This is descriptive only. Ultralytics may select best.pt using
    its internal fitness calculation.
    """

    if not results_csv.exists():
        return {
            "available": False,
            "reason": (
                "results.csv was not found."
            ),
        }

    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        return {
            "available": False,
            "reason": (
                "results.csv contains no data rows."
            ),
        }

    metric_key = (
        "metrics/mAP50-95(B)"
    )

    if metric_key not in rows[0]:
        return {
            "available": False,
            "reason": (
                f"Column '{metric_key}' "
                "was not found."
            ),
        }

    valid_rows: list[
        dict[str, str]
    ] = []

    for row in rows:
        try:
            float(row[metric_key])

        except (
            TypeError,
            ValueError,
            KeyError,
        ):
            continue

        valid_rows.append(row)

    if not valid_rows:
        return {
            "available": False,
            "reason": (
                "No numeric values were found in "
                f"column '{metric_key}'."
            ),
        }

    highest_map_row = max(
        valid_rows,
        key=lambda row: float(
            row[metric_key]
        ),
    )

    desired_columns = [
        "epoch",
        "time",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "lr/pg0",
        "lr/pg1",
        "lr/pg2",
    ]

    summary: dict[str, Any] = {
        "available": True,
        "selection_metric": metric_key,
        "note": (
            "This is the epoch with the highest recorded validation "
            "mAP@0.5:0.95. It is not necessarily the exact epoch used "
            "by Ultralytics to select best.pt."
        ),
    }

    for column in desired_columns:
        if column not in highest_map_row:
            continue

        value = highest_map_row[
            column
        ]

        try:
            numeric_value = float(
                value
            )

            if column == "epoch":
                summary[column] = int(
                    numeric_value
                )
            else:
                summary[column] = (
                    numeric_value
                )

        except (
            TypeError,
            ValueError,
        ):
            summary[column] = value

    return summary


def save_resolved_configuration(
    output_path: Path,
    configuration: dict[str, Any],
) -> None:
    """Save the exact resolved journal configuration."""

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            configuration,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def save_json(
    output_path: Path,
    content: dict[str, Any],
) -> None:
    """Save a dictionary as formatted JSON."""

    output_path.write_text(
        json.dumps(
            content,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def format_paths(
    paths: list[Path],
) -> str:
    """Format one or more paths for console output."""

    return ", ".join(
        str(path)
        for path in paths
    )


def print_experiment_summary(
    configuration: dict[str, Any],
    config_path: Path,
    model_source: str,
    data_yaml: Path,
    dataset_root: Path,
    split_paths: dict[str, list[Path]],
    split_report_record: dict[str, Any],
    project_directory: Path,
    training_arguments: dict[str, Any],
) -> None:
    """Print the resolved settings before training."""

    experiment = configuration[
        "experiment"
    ]

    print("=" * 80)
    print("JOURNAL YOLO EXPERIMENT")
    print("=" * 80)
    print(
        f"Configuration     : {config_path}"
    )
    print(
        f"Description       : "
        f"{experiment['description']}"
    )
    print(
        f"Experiment group  : "
        f"{experiment['group']}"
    )
    print(
        f"Experiment name   : "
        f"{experiment['name']}"
    )
    print(
        f"Model source      : {model_source}"
    )
    print(
        f"Dataset YAML      : {data_yaml}"
    )
    print(
        f"Dataset root      : {dataset_root}"
    )
    print(
        "Training path     : "
        f"{format_paths(split_paths['train'])}"
    )
    print(
        "Validation path   : "
        f"{format_paths(split_paths['val'])}"
    )
    print(
        "Testing path      : "
        f"{format_paths(split_paths['test'])}"
    )
    print(
        "Split report      : "
        f"{split_report_record.get('path')}"
    )
    print(
        "Split report SHA  : "
        f"{split_report_record.get('sha256')}"
    )
    print(
        f"Image size        : "
        f"{training_arguments['imgsz']}"
    )
    print(
        f"Epochs            : "
        f"{training_arguments['epochs']}"
    )
    print(
        f"Batch size        : "
        f"{training_arguments['batch']}"
    )
    print(
        f"Optimizer         : "
        f"{training_arguments['optimizer']}"
    )
    print(
        f"Initial LR        : "
        f"{training_arguments['lr0']}"
    )
    print(
        f"Final LR factor   : "
        f"{training_arguments['lrf']}"
    )
    print(
        f"Momentum          : "
        f"{training_arguments['momentum']}"
    )
    print(
        f"Weight decay      : "
        f"{training_arguments['weight_decay']}"
    )
    print(
        f"Patience          : "
        f"{training_arguments['patience']}"
    )
    print(
        f"Seed              : "
        f"{training_arguments['seed']}"
    )
    print(
        f"Deterministic     : "
        f"{training_arguments['deterministic']}"
    )
    print(
        f"Device            : "
        f"{training_arguments['device']}"
    )
    print(
        f"Workers           : "
        f"{training_arguments['workers']}"
    )
    print(
        f"AMP               : "
        f"{training_arguments['amp']}"
    )
    print(
        f"Cache             : "
        f"{training_arguments['cache']}"
    )
    print(
        "Expected output   : "
        f"{project_directory / str(experiment['name'])}"
    )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Load, validate, and run one configured experiment."""

    arguments = parse_arguments()

    repository_root = (
        resolve_repository_root()
    )

    config_path = (
        arguments.config.expanduser()
    )

    if not config_path.is_absolute():
        config_path = (
            repository_root / config_path
        )

    config_path = (
        config_path.resolve()
    )

    configuration = load_yaml_file(
        config_path
    )

    validate_experiment_configuration(
        configuration
    )

    if arguments.smoke_test:
        configuration = (
            apply_smoke_test_overrides(
                configuration
            )
        )

    experiment = configuration[
        "experiment"
    ]

    reproducibility = configuration[
        "reproducibility"
    ]

    runtime = configuration[
        "runtime"
    ]

    data_yaml = resolve_from_repository(
        configuration["dataset"]["data"],
        repository_root,
    )

    (
        _,
        dataset_root,
        class_names,
        split_paths,
        split_report_record,
    ) = load_and_validate_data_yaml(
        data_yaml
    )

    model_source = resolve_model_source(
        str(
            configuration["model"]["weights"]
        ),
        repository_root,
    )

    seed = int(
        reproducibility["seed"]
    )

    deterministic = bool(
        reproducibility["deterministic"]
    )

    set_random_seeds(seed)

    configure_determinism(
        deterministic
    )

    requested_device = str(
        runtime["device"]
    ).strip().lower()

    cpu_device_names = {
        "cpu",
        "mps",
    }

    requests_cuda = (
        requested_device
        not in cpu_device_names
        and requested_device != ""
    )

    if (
        requests_cuda
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "A CUDA device was requested in the configuration, "
            "but torch.cuda.is_available() returned False."
        )

    project_directory = (
        repository_root
        / "journal_project"
        / "experiments"
        / str(experiment["group"])
    ).resolve()

    project_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_arguments = (
        build_training_arguments(
            configuration=configuration,
            data_yaml=data_yaml,
            project_directory=(
                project_directory
            ),
        )
    )

    print_experiment_summary(
        configuration=configuration,
        config_path=config_path,
        model_source=model_source,
        data_yaml=data_yaml,
        dataset_root=dataset_root,
        split_paths=split_paths,
        split_report_record=(
            split_report_record
        ),
        project_directory=(
            project_directory
        ),
        training_arguments=(
            training_arguments
        ),
    )

    print("\nDataset classes:")

    for (
        class_id,
        class_name,
    ) in class_names.items():
        print(
            f"  {class_id}: {class_name}"
        )

    if split_report_record.get(
        "available"
    ):
        report = split_report_record.get(
            "report",
            {},
        )

        report_splits = report.get(
            "splits",
            {},
        )

        print(
            "\nFrozen split summary:"
        )

        for split_name in (
            "train",
            "val",
            "test",
        ):
            split_summary = (
                report_splits.get(
                    split_name,
                    {},
                )
            )

            image_count = (
                split_summary.get(
                    "images",
                    "unknown",
                )
            )

            object_count = (
                split_summary.get(
                    "objects",
                    "unknown",
                )
            )

            print(
                f"  {split_name:5s}: "
                f"{image_count} images, "
                f"{object_count} objects"
            )

        configuration_summary = (
            report.get(
                "configuration",
                {},
            )
        )

        temporal_verified = (
            configuration_summary.get(
                "temporal_separation_verified",
                "unknown",
            )
        )

        discarded_frames = (
            configuration_summary.get(
                "discarded_boundary_frames",
                "unknown",
            )
        )

        minimum_distance = (
            configuration_summary.get(
                "minimum_expected_frame_distance",
                "unknown",
            )
        )

        print(
            "  Temporal verification: "
            f"{temporal_verified}"
        )

        print(
            "  Boundary frames removed: "
            f"{discarded_frames}"
        )

        print(
            "  Minimum frame distance: "
            f"{minimum_distance}"
        )

    else:
        print(
            "\nWarning: split_report.json was not found. "
            "Training can proceed, but the frozen split cannot "
            "be cryptographically recorded."
        )

    print("\nBuilding model...")

    model = YOLO(
        model_source
    )

    model.info(
        verbose=True
    )

    print("\nStarting training...")

    training_start = (
        datetime.now()
    )

    results = model.train(
        **training_arguments
    )

    training_end = (
        datetime.now()
    )

    results_directory = Path(
        results.save_dir
    ).resolve()

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment_record = (
        build_environment_record(
            repository_root=(
                repository_root
            ),
            config_path=config_path,
            data_yaml=data_yaml,
            dataset_root=dataset_root,
            split_paths=split_paths,
            split_report_record=(
                split_report_record
            ),
            model_source=model_source,
            training_arguments=(
                training_arguments
            ),
        )
    )

    environment_record[
        "training_started_at"
    ] = training_start.isoformat(
        timespec="seconds"
    )

    environment_record[
        "training_finished_at"
    ] = training_end.isoformat(
        timespec="seconds"
    )

    environment_record[
        "total_training_seconds"
    ] = (
        training_end
        - training_start
    ).total_seconds()

    environment_record[
        "results_directory"
    ] = str(
        results_directory
    )

    save_json(
        results_directory
        / "environment_record.json",
        environment_record,
    )

    save_resolved_configuration(
        results_directory
        / "journal_config.yaml",
        configuration,
    )

    shutil.copy2(
        config_path,
        results_directory
        / "source_config.yaml",
    )

    if split_report_record.get(
        "available"
    ):
        source_split_report = Path(
            str(
                split_report_record[
                    "path"
                ]
            )
        )

        shutil.copy2(
            source_split_report,
            results_directory
            / "dataset_split_report.json",
        )

    highest_map_epoch_summary = (
        read_highest_map_epoch_summary(
            results_directory
            / "results.csv"
        )
    )

    compact_split_report = {
        key: value
        for key, value
        in split_report_record.items()
        if key != "report"
    }

    experiment_summary = {
        "experiment_group": str(
            experiment["group"]
        ),
        "experiment_name": str(
            experiment["name"]
        ),
        "description": str(
            experiment["description"]
        ),
        "model_source": (
            model_source
        ),
        "data_yaml": str(
            data_yaml
        ),
        "data_yaml_sha256": (
            calculate_sha256(
                data_yaml
            )
        ),
        "dataset_root": str(
            dataset_root
        ),
        "dataset_split_paths": {
            split_name: [
                str(path)
                for path in paths
            ]
            for split_name, paths
            in split_paths.items()
        },
        "dataset_split_report": (
            compact_split_report
        ),
        "class_names": (
            class_names
        ),
        "training_started_at": (
            training_start.isoformat(
                timespec="seconds"
            )
        ),
        "training_finished_at": (
            training_end.isoformat(
                timespec="seconds"
            )
        ),
        "total_training_seconds": (
            training_end
            - training_start
        ).total_seconds(),
        "results_directory": str(
            results_directory
        ),
        "best_checkpoint": str(
            results_directory
            / "weights"
            / "best.pt"
        ),
        "last_checkpoint": str(
            results_directory
            / "weights"
            / "last.pt"
        ),
        "highest_map50_95_epoch_summary": (
            highest_map_epoch_summary
        ),
    }

    save_json(
        results_directory
        / "experiment_summary.json",
        experiment_summary,
    )

    print(
        "\n" + "=" * 80
    )
    print(
        "TRAINING FINISHED"
    )
    print(
        "=" * 80
    )
    print(
        "Results directory : "
        f"{results_directory}"
    )
    print(
        "Best checkpoint   : "
        f"{results_directory / 'weights' / 'best.pt'}"
    )
    print(
        "Last checkpoint   : "
        f"{results_directory / 'weights' / 'last.pt'}"
    )
    print(
        "Training duration : "
        f"{experiment_summary['total_training_seconds']:.2f} "
        "seconds"
    )

    if highest_map_epoch_summary.get(
        "available"
    ):
        highest_epoch = (
            highest_map_epoch_summary.get(
                "epoch",
                "unknown",
            )
        )

        map50 = (
            highest_map_epoch_summary.get(
                "metrics/mAP50(B)",
                "unknown",
            )
        )

        map50_95 = (
            highest_map_epoch_summary.get(
                "metrics/mAP50-95(B)",
                "unknown",
            )
        )

        precision = (
            highest_map_epoch_summary.get(
                "metrics/precision(B)",
                "unknown",
            )
        )

        recall = (
            highest_map_epoch_summary.get(
                "metrics/recall(B)",
                "unknown",
            )
        )

        print(
            f"Highest-mAP epoch : {highest_epoch}"
        )
        print(
            f"mAP@0.5           : {map50}"
        )
        print(
            f"mAP@0.5:0.95      : {map50_95}"
        )
        print(
            f"Precision         : {precision}"
        )
        print(
            f"Recall            : {recall}"
        )

    else:
        print(
            "Highest-mAP summary could not be generated: "
            f"{highest_map_epoch_summary.get('reason', 'unknown')}"
        )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nTraining was interrupted by the user."
        )
        sys.exit(130)

    except Exception as error:
        print(
            "\nEXPERIMENT FAILED"
        )
        print(
            f"{type(error).__name__}: {error}"
        )
        raise