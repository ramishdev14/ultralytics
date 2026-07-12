"""
Train the plain YOLO11s baseline for the journal study.

Run modes:
    Smoke test:
        python journal_project/scripts/02_train_yolo11s_baseline.py --smoke-test

    Full training:
        python journal_project/scripts/02_train_yolo11s_baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


DEFAULT_SEED = 42


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the YOLO11s baseline using fixed journal settings."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../dataset/data.yaml"),
        help="Path to data.yaml. Default: ../dataset/data.yaml",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short 3-epoch architecture and pipeline test.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device, for example 0 or cpu. Default: 0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers. Default: 4",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )

    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    """Set major Python, NumPy, and PyTorch random seeds."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_git_command(arguments: list[str], repository: Path) -> str:
    """Run a Git command and return a safe string result."""

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unavailable"


def load_and_validate_data_yaml(data_yaml: Path) -> dict[str, Any]:
    """Validate that the dataset YAML and referenced folders exist."""

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError("data.yaml must contain a YAML dictionary.")

    required_keys = {"path", "train", "val", "names"}

    missing = required_keys.difference(content)

    if missing:
        raise ValueError(
            f"data.yaml is missing required fields: {sorted(missing)}"
        )

    dataset_root = Path(str(content["path"])).expanduser()

    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    dataset_root = dataset_root.resolve()

    train_path = dataset_root / str(content["train"])
    val_path = dataset_root / str(content["val"])

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not train_path.exists():
        raise FileNotFoundError(f"Training images not found: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation images not found: {val_path}")

    names = content["names"]

    if isinstance(names, list):
        number_of_classes = len(names)
    elif isinstance(names, dict):
        number_of_classes = len(names)
    else:
        raise ValueError("'names' must be a list or dictionary.")

    declared_nc = content.get("nc")

    if declared_nc is not None and int(declared_nc) != number_of_classes:
        raise ValueError(
            f"nc={declared_nc}, but {number_of_classes} names were found."
        )

    return content


def create_environment_record(
    output_path: Path,
    repository_root: Path,
    data_yaml: Path,
    configuration: dict[str, Any],
) -> None:
    """Save a reproducibility record before training."""

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    gpu_memory_gb = None

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu_memory_gb = round(properties.total_memory / 1024**3, 2)

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_source": str(Path(ultralytics.__file__).resolve()),
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
        "data_yaml": str(data_yaml),
        "training_configuration": configuration,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]

    data_yaml = arguments.data.expanduser()

    if not data_yaml.is_absolute():
        data_yaml = (repository_root / data_yaml).resolve()
    else:
        data_yaml = data_yaml.resolve()

    load_and_validate_data_yaml(data_yaml)

    set_random_seeds(arguments.seed)

    if not torch.cuda.is_available() and arguments.device != "cpu":
        raise RuntimeError(
            "CUDA is unavailable, but a GPU device was requested."
        )

    if arguments.smoke_test:
        epochs = 3
        patience = 3
        experiment_name = "yolo11s_baseline_smoke"
    else:
        epochs = 100
        patience = 20
        experiment_name = "yolo11s_baseline_full"

    project_directory = repository_root / "journal_project" / "experiments"

    training_configuration: dict[str, Any] = {
        "model": "yolo11s.pt",
        "data": str(data_yaml),
        "imgsz": 640,
        "epochs": epochs,
        "batch": 16,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "momentum": 0.93,
        "weight_decay": 5e-4,
        "patience": patience,
        "seed": arguments.seed,
        "deterministic": True,
        "device": arguments.device,
        "workers": arguments.workers,
        "project": str(project_directory),
        "name": experiment_name,
        "exist_ok": False,
        "pretrained": True,
        "amp": True,
        "cache": False,
        "plots": True,
        "save": True,
        "save_period": -1,
        "verbose": True,
    }

    metadata_path = (
        project_directory
        / experiment_name
        / "environment_record.json"
    )

    create_environment_record(
        output_path=metadata_path,
        repository_root=repository_root,
        data_yaml=data_yaml,
        configuration=training_configuration,
    )

    print("=" * 78)
    print("YOLO11S BASELINE TRAINING")
    print("=" * 78)
    print(f"Repository root : {repository_root}")
    print(f"Dataset YAML    : {data_yaml}")
    print(f"Experiment      : {experiment_name}")
    print(f"Epochs          : {epochs}")
    print(f"Batch size      : 16")
    print(f"Image size      : 640")
    print(f"Device          : {arguments.device}")
    print(f"Workers         : {arguments.workers}")
    print(f"Seed            : {arguments.seed}")
    print(f"Output          : {project_directory / experiment_name}")
    print("=" * 78)

    model = YOLO("yolo11s.pt")

    model.info(verbose=True)

    results = model.train(**training_configuration)

    print("\n" + "=" * 78)
    print("TRAINING FINISHED")
    print("=" * 78)
    print(f"Results directory: {results.save_dir}")
    print(f"Best checkpoint  : {Path(results.save_dir) / 'weights' / 'best.pt'}")
    print(f"Last checkpoint  : {Path(results.save_dir) / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by the user.")
        sys.exit(130)
    except Exception as error:
        print("\nBASELINE TRAINING FAILED")
        print(f"{type(error).__name__}: {error}")
        raise