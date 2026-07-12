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
        raise ValueError(f"Invalid YAML syntax in {path}: {error}") from error

    if not isinstance(content, dict):
        raise ValueError(f"YAML file must contain a dictionary: {path}")

    return content


def require_section(
    configuration: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Retrieve a required dictionary section."""

    section = configuration.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(
            f"Configuration section '{section_name}' is missing or is not "
            "a dictionary."
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

    experiment = require_section(configuration, "experiment")
    model = require_section(configuration, "model")
    dataset = require_section(configuration, "dataset")
    training = require_section(configuration, "training")
    reproducibility = require_section(configuration, "reproducibility")
    runtime = require_section(configuration, "runtime")
    output = require_section(configuration, "output")

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
        raise ValueError("experiment.group cannot be empty.")

    if not str(experiment["name"]).strip():
        raise ValueError("experiment.name cannot be empty.")

    if int(training["epochs"]) <= 0:
        raise ValueError("training.epochs must be greater than zero.")

    if int(training["batch"]) == 0:
        raise ValueError("training.batch cannot be zero.")

    if int(training["imgsz"]) <= 0:
        raise ValueError("training.imgsz must be greater than zero.")

    if "workers" in training:
        raise ValueError(
        "The 'workers' setting belongs under the runtime section, "
        "not the training section."
    )

    if int(runtime["workers"]) < 0:
        raise ValueError("runtime.workers cannot be negative.")

    if float(training["lr0"]) <= 0:
        raise ValueError("training.lr0 must be greater than zero.")

    if float(training["lrf"]) <= 0:
        raise ValueError("training.lrf must be greater than zero.")

    if int(reproducibility["seed"]) < 0:
        raise ValueError("reproducibility.seed cannot be negative.")


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

    if not (repository_root / "pyproject.toml").exists():
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
    Resolve a local model path or retain an Ultralytics model identifier.

    Examples retained as identifiers:
        yolo11s.pt
        yolo11s.yaml

    Examples resolved as local paths:
        journal_project/models/yolo11_ca.yaml
    """

    model_value = str(model_value).strip()

    if not model_value:
        raise ValueError("model.weights cannot be empty.")

    candidate = Path(model_value).expanduser()

    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(
                f"Configured model file was not found: {candidate}"
            )

        return str(candidate.resolve())

    repository_candidate = repository_root / candidate

    if repository_candidate.exists():
        return str(repository_candidate.resolve())

    # Allow official Ultralytics names such as yolo11s.pt to be downloaded.
    if candidate.parent == Path("."):
        return model_value

    raise FileNotFoundError(
        "Configured local model file was not found. "
        f"Checked: {repository_candidate.resolve()}"
    )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def normalize_class_names(
    names: list[str] | dict[int | str, str],
) -> dict[int, str]:
    """Convert dataset class names into an integer-keyed dictionary."""

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
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Dataset class-name dictionary keys must be integers."
            ) from error

    else:
        raise ValueError(
            "The data.yaml 'names' field must be a list or dictionary."
        )

    expected_ids = list(range(len(class_names)))

    if sorted(class_names) != expected_ids:
        raise ValueError(
            "Dataset class IDs must begin at zero and be contiguous. "
            f"Found: {sorted(class_names)}"
        )

    return class_names


def load_and_validate_data_yaml(
    data_yaml: Path,
) -> tuple[dict[str, Any], Path, dict[int, str]]:
    """Validate the dataset YAML and its referenced directories."""

    content = load_yaml_file(data_yaml)

    required_keys = {"path", "train", "val", "names"}
    missing_keys = required_keys.difference(content)

    if missing_keys:
        raise ValueError(
            f"Dataset YAML is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    dataset_root = Path(str(content["path"])).expanduser()

    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    dataset_root = dataset_root.resolve()

    train_path = dataset_root / str(content["train"])
    val_path = dataset_root / str(content["val"])

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root was not found: {dataset_root}"
        )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training image path was not found: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation image path was not found: {val_path}"
        )

    class_names = normalize_class_names(content["names"])

    declared_nc = content.get("nc")

    if declared_nc is not None and int(declared_nc) != len(class_names):
        raise ValueError(
            f"data.yaml declares nc={declared_nc}, but "
            f"{len(class_names)} class names were found."
        )

    return content, dataset_root, class_names


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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

    for device_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_index)

        devices.append(
            {
                "index": device_index,
                "name": properties.name,
                "total_memory_gb": round(
                    properties.total_memory / 1024**3,
                    2,
                ),
                "compute_capability": (
                    f"{properties.major}.{properties.minor}"
                ),
            }
        )

    return devices


def build_environment_record(
    repository_root: Path,
    config_path: Path,
    data_yaml: Path,
    model_source: str,
    training_arguments: dict[str, Any],
) -> dict[str, Any]:
    """Build a reproducibility and environment record."""

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": get_gpu_information(),
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_source": str(
            Path(ultralytics.__file__).resolve()
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
        "config_path": str(config_path),
        "data_yaml": str(data_yaml),
        "model_source": model_source,
        "training_arguments": training_arguments,
    }


# ---------------------------------------------------------------------------
# Configuration conversion
# ---------------------------------------------------------------------------


def apply_smoke_test_overrides(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Create an in-memory three-epoch smoke-test configuration."""

    smoke_configuration = deepcopy(configuration)

    experiment = smoke_configuration["experiment"]
    training = smoke_configuration["training"]

    experiment["name"] = f"{experiment['name']}_smoke"
    experiment["description"] = (
        f"{experiment['description']} [three-epoch smoke test]"
    )

    training["epochs"] = 3
    training["patience"] = 3

    return smoke_configuration


def build_training_arguments(
    configuration: dict[str, Any],
    data_yaml: Path,
    project_directory: Path,
) -> dict[str, Any]:
    """Convert the journal YAML configuration to Ultralytics arguments."""

    experiment = configuration["experiment"]
    training = configuration["training"]
    reproducibility = configuration["reproducibility"]
    runtime = configuration["runtime"]
    output = configuration["output"]

    training_arguments: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": int(training["imgsz"]),
        "epochs": int(training["epochs"]),
        "batch": int(training["batch"]),
        "optimizer": str(training["optimizer"]),
        "lr0": float(training["lr0"]),
        "lrf": float(training["lrf"]),
        "momentum": float(training["momentum"]),
        "weight_decay": float(training["weight_decay"]),
        "patience": int(training["patience"]),
        "seed": int(reproducibility["seed"]),
        "deterministic": bool(reproducibility["deterministic"]),
        "device": str(runtime["device"]),
        "workers": int(runtime["workers"]),
        "amp": bool(runtime["amp"]),
        "cache": runtime["cache"],
        "project": str(project_directory),
        "name": str(experiment["name"]),
        "exist_ok": bool(output.get("exist_ok", False)),
        "pretrained": bool(configuration["model"].get("pretrained", True)),
        "plots": bool(output["plots"]),
        "save": bool(output["save"]),
        "save_period": int(output["save_period"]),
        "verbose": bool(output.get("verbose", True)),
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
    }

    for key in optional_training_keys:
        if key in training:
            training_arguments[key] = training[key]

    return training_arguments


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------


def read_best_epoch_summary(
    results_csv: Path,
) -> dict[str, Any]:
    """
    Read results.csv and identify the epoch with the highest mAP@0.5:0.95.

    This is used only for the generated experiment summary.
    """

    if not results_csv.exists():
        return {
            "available": False,
            "reason": "results.csv was not found.",
        }

    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return {
            "available": False,
            "reason": "results.csv contains no data rows.",
        }

    metric_key = "metrics/mAP50-95(B)"

    if metric_key not in rows[0]:
        return {
            "available": False,
            "reason": f"Column '{metric_key}' was not found.",
        }

    best_row = max(
        rows,
        key=lambda row: float(row[metric_key]),
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
    ]

    summary: dict[str, Any] = {
        "available": True,
        "selection_metric": metric_key,
    }

    for column in desired_columns:
        if column not in best_row:
            continue

        value = best_row[column]

        try:
            numeric_value = float(value)

            if column == "epoch":
                summary[column] = int(numeric_value)
            else:
                summary[column] = numeric_value

        except ValueError:
            summary[column] = value

    return summary


def save_resolved_configuration(
    output_path: Path,
    configuration: dict[str, Any],
) -> None:
    """Save the exact resolved journal configuration used for the run."""

    with output_path.open("w", encoding="utf-8") as file:
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
        json.dumps(content, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_experiment_summary(
    configuration: dict[str, Any],
    config_path: Path,
    model_source: str,
    data_yaml: Path,
    dataset_root: Path,
    project_directory: Path,
    training_arguments: dict[str, Any],
) -> None:
    """Print the resolved experiment settings before training."""

    experiment = configuration["experiment"]

    print("=" * 80)
    print("JOURNAL YOLO EXPERIMENT")
    print("=" * 80)
    print(f"Configuration     : {config_path}")
    print(f"Description       : {experiment['description']}")
    print(f"Experiment group  : {experiment['group']}")
    print(f"Experiment name   : {experiment['name']}")
    print(f"Model source      : {model_source}")
    print(f"Dataset YAML      : {data_yaml}")
    print(f"Dataset root      : {dataset_root}")
    print(f"Image size        : {training_arguments['imgsz']}")
    print(f"Epochs            : {training_arguments['epochs']}")
    print(f"Batch size        : {training_arguments['batch']}")
    print(f"Optimizer         : {training_arguments['optimizer']}")
    print(f"Initial LR        : {training_arguments['lr0']}")
    print(f"Final LR factor   : {training_arguments['lrf']}")
    print(f"Momentum          : {training_arguments['momentum']}")
    print(f"Weight decay      : {training_arguments['weight_decay']}")
    print(f"Patience          : {training_arguments['patience']}")
    print(f"Seed              : {training_arguments['seed']}")
    print(f"Deterministic     : {training_arguments['deterministic']}")
    print(f"Device            : {training_arguments['device']}")
    print(f"Workers           : {training_arguments['workers']}")
    print(f"AMP               : {training_arguments['amp']}")
    print(f"Cache             : {training_arguments['cache']}")
    print(
        f"Expected output   : "
        f"{project_directory / str(experiment['name'])}"
    )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Load, validate, and run one configured experiment."""

    arguments = parse_arguments()
    repository_root = resolve_repository_root()

    config_path = arguments.config.expanduser()

    if not config_path.is_absolute():
        config_path = repository_root / config_path

    config_path = config_path.resolve()

    configuration = load_yaml_file(config_path)
    validate_experiment_configuration(configuration)

    if arguments.smoke_test:
        configuration = apply_smoke_test_overrides(configuration)

    experiment = configuration["experiment"]
    reproducibility = configuration["reproducibility"]
    runtime = configuration["runtime"]

    data_yaml = resolve_from_repository(
        configuration["dataset"]["data"],
        repository_root,
    )

    _, dataset_root, class_names = load_and_validate_data_yaml(data_yaml)

    model_source = resolve_model_source(
        str(configuration["model"]["weights"]),
        repository_root,
    )

    seed = int(reproducibility["seed"])
    set_random_seeds(seed)

    requested_device = str(runtime["device"]).strip().lower()

    if requested_device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA device was requested in the configuration, but "
            "torch.cuda.is_available() returned False."
        )

    project_directory = (
        repository_root
        / "journal_project"
        / "experiments"
        / str(experiment["group"])
    ).resolve()

    project_directory.mkdir(parents=True, exist_ok=True)

    training_arguments = build_training_arguments(
        configuration=configuration,
        data_yaml=data_yaml,
        project_directory=project_directory,
    )

    print_experiment_summary(
        configuration=configuration,
        config_path=config_path,
        model_source=model_source,
        data_yaml=data_yaml,
        dataset_root=dataset_root,
        project_directory=project_directory,
        training_arguments=training_arguments,
    )

    print("\nDataset classes:")

    for class_id, class_name in class_names.items():
        print(f"  {class_id}: {class_name}")

    print("\nBuilding model...")

    model = YOLO(model_source)
    model.info(verbose=True)

    print("\nStarting training...")

    training_start = datetime.now()
    results = model.train(**training_arguments)
    training_end = datetime.now()

    results_directory = Path(results.save_dir).resolve()
    results_directory.mkdir(parents=True, exist_ok=True)

    environment_record = build_environment_record(
        repository_root=repository_root,
        config_path=config_path,
        data_yaml=data_yaml,
        model_source=model_source,
        training_arguments=training_arguments,
    )

    environment_record["training_started_at"] = training_start.isoformat(
        timespec="seconds"
    )

    environment_record["training_finished_at"] = training_end.isoformat(
        timespec="seconds"
    )

    environment_record["total_training_seconds"] = (
        training_end - training_start
    ).total_seconds()

    environment_record["results_directory"] = str(results_directory)

    save_json(
        results_directory / "environment_record.json",
        environment_record,
    )

    save_resolved_configuration(
        results_directory / "journal_config.yaml",
        configuration,
    )

    shutil.copy2(
        config_path,
        results_directory / "source_config.yaml",
    )

    best_epoch_summary = read_best_epoch_summary(
        results_directory / "results.csv"
    )

    experiment_summary = {
        "experiment_group": str(experiment["group"]),
        "experiment_name": str(experiment["name"]),
        "description": str(experiment["description"]),
        "model_source": model_source,
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "training_started_at": training_start.isoformat(
            timespec="seconds"
        ),
        "training_finished_at": training_end.isoformat(
            timespec="seconds"
        ),
        "total_training_seconds": (
            training_end - training_start
        ).total_seconds(),
        "results_directory": str(results_directory),
        "best_checkpoint": str(
            results_directory / "weights" / "best.pt"
        ),
        "last_checkpoint": str(
            results_directory / "weights" / "last.pt"
        ),
        "best_epoch_summary": best_epoch_summary,
    }

    save_json(
        results_directory / "experiment_summary.json",
        experiment_summary,
    )

    print("\n" + "=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)
    print(f"Results directory : {results_directory}")
    print(
        f"Best checkpoint   : "
        f"{results_directory / 'weights' / 'best.pt'}"
    )
    print(
        f"Last checkpoint   : "
        f"{results_directory / 'weights' / 'last.pt'}"
    )
    print(
        f"Training duration : "
        f"{experiment_summary['total_training_seconds']:.2f} seconds"
    )

    if best_epoch_summary.get("available"):
        print(
            f"Best epoch        : "
            f"{best_epoch_summary.get('epoch', 'unknown')}"
        )
        print(
            f"Best mAP@0.5      : "
            f"{best_epoch_summary.get('metrics/mAP50(B)', 'unknown')}"
        )
        print(
            f"Best mAP@0.5:0.95 : "
            f"{best_epoch_summary.get('metrics/mAP50-95(B)', 'unknown')}"
        )
        print(
            f"Precision         : "
            f"{best_epoch_summary.get('metrics/precision(B)', 'unknown')}"
        )
        print(
            f"Recall            : "
            f"{best_epoch_summary.get('metrics/recall(B)', 'unknown')}"
        )

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nTraining was interrupted by the user.")
        sys.exit(130)

    except Exception as error:
        print("\nEXPERIMENT FAILED")
        print(f"{type(error).__name__}: {error}")
        raise