"""
Evaluate a trained Ultralytics YOLO model on the untouched test split.

Example
-------
python journal_project/scripts/evaluate.py ^
    --weights journal_project/experiments/baseline/yolo11s-2/weights/best.pt ^
    --data ../dataset_grouped_final/data.yaml ^
    --name yolo11s_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Ultralytics YOLO detection model "
            "on the dataset test split."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained best.pt checkpoint.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the dataset data.yaml file.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="test_evaluation",
        help="Name of the test-evaluation output directory.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Evaluation device, for example 0 or cpu.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help=(
            "Confidence threshold used during metric calculation. "
            "Ultralytics commonly uses a low threshold for evaluation."
        ),
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold used for non-maximum suppression.",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image.",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 evaluation where supported.",
    )

    return parser.parse_args()


def resolve_repository_root() -> Path:
    """
    Resolve the cloned Ultralytics repository root.

    Expected location:
        <repository>/journal_project/scripts/evaluate.py
    """

    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]

    if not (repository_root / "pyproject.toml").exists():
        raise RuntimeError(
            "Could not identify the Ultralytics repository root. "
            f"Resolved candidate: {repository_root}"
        )

    return repository_root


def resolve_path(
    path: Path,
    repository_root: Path,
) -> Path:
    """Resolve a path relative to the repository root."""

    path = path.expanduser()

    if not path.is_absolute():
        path = repository_root / path

    return path.resolve()


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and validate a YAML file."""

    if not path.exists():
        raise FileNotFoundError(
            f"YAML file was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a YAML file but received: {path}"
        )

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


def resolve_dataset_root(
    data_yaml: Path,
    data_content: dict[str, Any],
) -> Path:
    """Resolve the dataset root declared by data.yaml."""

    if "path" not in data_content:
        raise ValueError(
            "Dataset YAML is missing the required 'path' field."
        )

    dataset_root = Path(
        str(data_content["path"])
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

    return dataset_root


def resolve_dataset_entry(
    dataset_root: Path,
    value: str | list[str],
) -> list[Path]:
    """Resolve a dataset split entry."""

    values = value if isinstance(value, list) else [value]

    resolved_paths: list[Path] = []

    for item in values:
        split_path = Path(
            str(item)
        ).expanduser()

        if not split_path.is_absolute():
            split_path = (
                dataset_root / split_path
            )

        resolved_paths.append(
            split_path.resolve()
        )

    return resolved_paths


def validate_dataset_yaml(
    data_yaml: Path,
) -> tuple[
    dict[str, Any],
    Path,
    list[Path],
]:
    """Validate that data.yaml contains a usable test split."""

    data_content = load_yaml_file(
        data_yaml
    )

    required_keys = {
        "path",
        "test",
        "names",
    }

    missing_keys = required_keys.difference(
        data_content
    )

    if missing_keys:
        raise ValueError(
            "Dataset YAML is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    dataset_root = resolve_dataset_root(
        data_yaml,
        data_content,
    )

    test_paths = resolve_dataset_entry(
        dataset_root,
        data_content["test"],
    )

    if not test_paths:
        raise ValueError(
            "The test split contains no configured paths."
        )

    for test_path in test_paths:
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test split path was not found: {test_path}"
            )

        if not test_path.is_dir():
            raise ValueError(
                f"Test split path is not a directory: {test_path}"
            )

    return (
        data_content,
        dataset_root,
        test_paths,
    )


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


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
    """Return information about visible CUDA devices."""

    gpu_information: list[dict[str, Any]] = []

    if not torch.cuda.is_available():
        return gpu_information

    for device_index in range(
        torch.cuda.device_count()
    ):
        properties = (
            torch.cuda.get_device_properties(
                device_index
            )
        )

        gpu_information.append(
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

    return gpu_information


def convert_metric_value(
    value: Any,
) -> Any:
    """Convert metric values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if hasattr(value, "tolist"):
        return value.tolist()

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def extract_overall_metrics(
    metrics: Any,
) -> dict[str, Any]:
    """Extract overall detection metrics from Ultralytics results."""

    box = getattr(
        metrics,
        "box",
        None,
    )

    if box is None:
        raise RuntimeError(
            "Ultralytics did not return box-detection metrics."
        )

    return {
        "precision": convert_metric_value(
            getattr(box, "mp", None)
        ),
        "recall": convert_metric_value(
            getattr(box, "mr", None)
        ),
        "map50": convert_metric_value(
            getattr(box, "map50", None)
        ),
        "map75": convert_metric_value(
            getattr(box, "map75", None)
        ),
        "map50_95": convert_metric_value(
            getattr(box, "map", None)
        ),
    }


def extract_per_class_metrics(
    metrics: Any,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Extract per-class precision, recall, and mAP metrics."""

    box = getattr(
        metrics,
        "box",
        None,
    )

    if box is None:
        return []

    precision_values = getattr(
        box,
        "p",
        [],
    )

    recall_values = getattr(
        box,
        "r",
        [],
    )

    map50_values = getattr(
        box,
        "ap50",
        [],
    )

    map_values = getattr(
        box,
        "ap",
        [],
    )

    results: list[dict[str, Any]] = []

    for class_id, class_name in class_names.items():
        precision = None
        recall = None
        map50 = None
        map50_95 = None

        if class_id < len(precision_values):
            precision = convert_metric_value(
                precision_values[class_id]
            )

        if class_id < len(recall_values):
            recall = convert_metric_value(
                recall_values[class_id]
            )

        if class_id < len(map50_values):
            map50 = convert_metric_value(
                map50_values[class_id]
            )

        if class_id < len(map_values):
            map50_95 = convert_metric_value(
                map_values[class_id]
            )

        results.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "map50": map50,
                "map50_95": map50_95,
            }
        )

    return results


def normalize_class_names(
    names: list[str] | dict[int | str, str],
) -> dict[int, str]:
    """Normalize class names into an integer-keyed dictionary."""

    if isinstance(names, list):
        return {
            class_id: str(class_name)
            for class_id, class_name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(class_id): str(class_name)
            for class_id, class_name in names.items()
        }

    raise ValueError(
        "The dataset 'names' field must be a list or dictionary."
    )


def save_json(
    path: Path,
    content: dict[str, Any],
) -> None:
    """Save JSON content."""

    path.write_text(
        json.dumps(
            content,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run test-set evaluation."""

    arguments = parse_arguments()

    repository_root = resolve_repository_root()

    weights_path = resolve_path(
        arguments.weights,
        repository_root,
    )

    data_yaml = resolve_path(
        arguments.data,
        repository_root,
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {weights_path}"
        )

    if not weights_path.is_file():
        raise ValueError(
            f"Model checkpoint is not a file: {weights_path}"
        )

    (
        data_content,
        dataset_root,
        test_paths,
    ) = validate_dataset_yaml(
        data_yaml
    )

    class_names = normalize_class_names(
        data_content["names"]
    )

    requested_device = (
        arguments.device.strip().lower()
    )

    if (
        requested_device not in {"cpu", "mps", ""}
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "A CUDA device was requested, but CUDA is unavailable."
        )

    output_project = (
        repository_root
        / "journal_project"
        / "results"
        / "test_evaluations"
    ).resolve()

    output_project.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("YOLO TEST-SET EVALUATION")
    print("=" * 80)
    print(f"Checkpoint        : {weights_path}")
    print(f"Dataset YAML      : {data_yaml}")
    print(f"Dataset root      : {dataset_root}")
    print(
        "Test split        : "
        + ", ".join(
            str(path)
            for path in test_paths
        )
    )
    print(f"Image size        : {arguments.imgsz}")
    print(f"Batch size        : {arguments.batch}")
    print(f"Device            : {arguments.device}")
    print(f"Workers           : {arguments.workers}")
    print(f"Output name       : {arguments.name}")
    print("=" * 80)

    model = YOLO(
        str(weights_path)
    )

    evaluation_start = datetime.now()

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        device=arguments.device,
        workers=arguments.workers,
        conf=arguments.conf,
        iou=arguments.iou,
        max_det=arguments.max_det,
        half=arguments.half,
        plots=True,
        save_json=True,
        project=str(output_project),
        name=arguments.name,
        exist_ok=False,
        verbose=True,
    )

    evaluation_end = datetime.now()

    results_directory = Path(
        metrics.save_dir
    ).resolve()

    overall_metrics = extract_overall_metrics(
        metrics
    )

    per_class_metrics = extract_per_class_metrics(
        metrics,
        class_names,
    )

    speed = getattr(
        metrics,
        "speed",
        {},
    )

    summary = {
        "evaluation_type": "test",
        "evaluation_started_at": (
            evaluation_start.isoformat(
                timespec="seconds"
            )
        ),
        "evaluation_finished_at": (
            evaluation_end.isoformat(
                timespec="seconds"
            )
        ),
        "evaluation_seconds": (
            evaluation_end
            - evaluation_start
        ).total_seconds(),
        "weights": str(weights_path),
        "weights_sha256": calculate_sha256(
            weights_path
        ),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": calculate_sha256(
            data_yaml
        ),
        "dataset_root": str(dataset_root),
        "test_paths": [
            str(path)
            for path in test_paths
        ],
        "results_directory": str(
            results_directory
        ),
        "settings": {
            "split": "test",
            "imgsz": arguments.imgsz,
            "batch": arguments.batch,
            "device": arguments.device,
            "workers": arguments.workers,
            "conf": arguments.conf,
            "iou": arguments.iou,
            "max_det": arguments.max_det,
            "half": arguments.half,
        },
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "speed_ms_per_image": {
            key: convert_metric_value(value)
            for key, value in speed.items()
        },
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": get_gpu_information(),
            "ultralytics_version": ultralytics.__version__,
            "ultralytics_source": str(
                Path(
                    ultralytics.__file__
                ).resolve()
            ),
            "git_commit": run_git_command(
                ["rev-parse", "HEAD"],
                repository_root,
            ),
            "git_status": run_git_command(
                ["status", "--short"],
                repository_root,
            ),
        },
    }

    save_json(
        results_directory
        / "test_summary.json",
        summary,
    )

    print("\n" + "=" * 80)
    print("TEST EVALUATION FINISHED")
    print("=" * 80)
    print(
        f"Results directory : {results_directory}"
    )
    print(
        f"Precision         : "
        f"{overall_metrics['precision']}"
    )
    print(
        f"Recall            : "
        f"{overall_metrics['recall']}"
    )
    print(
        f"mAP@0.5           : "
        f"{overall_metrics['map50']}"
    )
    print(
        f"mAP@0.75          : "
        f"{overall_metrics['map75']}"
    )
    print(
        f"mAP@0.5:0.95      : "
        f"{overall_metrics['map50_95']}"
    )

    print("\nPer-class test results:")

    for class_result in per_class_metrics:
        print(
            f"{class_result['class_name']:15s} "
            f"P={class_result['precision']}  "
            f"R={class_result['recall']}  "
            f"mAP50={class_result['map50']}  "
            f"mAP50-95={class_result['map50_95']}"
        )

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nEvaluation was interrupted by the user."
        )
        sys.exit(130)

    except Exception as error:
        print("\nTEST EVALUATION FAILED")
        print(
            f"{type(error).__name__}: {error}"
        )
        raise