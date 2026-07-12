"""
Audit a YOLO-format object detection dataset.

Checks:
1. data.yaml structure and class definitions.
2. Image-label filename matching.
3. Corrupted or unreadable images.
4. Empty label files.
5. YOLO label format:
       class_id x_center y_center width height
6. Valid class IDs.
7. Normalized coordinate ranges.
8. Boxes extending outside image boundaries.
9. Duplicate image stems.
10. Per-class instance counts.
11. Train/validation statistics.

Outputs:
- dataset_audit_report.json
- class_statistics.csv
- split_statistics.csv
- missing_labels.txt
- orphan_labels.txt
- invalid_labels.txt
- empty_labels.txt
- corrupted_images.txt
- duplicate_image_stems.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class SplitStatistics:
    """Summary statistics for one dataset split."""

    split: str
    image_count: int = 0
    label_file_count: int = 0
    matched_pairs: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    corrupted_images: int = 0
    empty_labels: int = 0
    invalid_label_files: int = 0
    valid_instances: int = 0
    invalid_instances: int = 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit a YOLO-format object detection dataset."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the dataset data.yaml file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory for audit reports. "
            "Default: <dataset_root>/audit_reports"
        ),
    )

    return parser.parse_args()


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load and validate the basic YAML structure."""

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml was not found: {yaml_path}")

    if not yaml_path.is_file():
        raise ValueError(f"The supplied YAML path is not a file: {yaml_path}")

    try:
        with yaml_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML syntax in {yaml_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("The data.yaml file must contain a YAML dictionary.")

    required_keys = {"path", "train", "val", "names"}
    missing_keys = required_keys.difference(data)

    if missing_keys:
        raise ValueError(
            f"Missing required data.yaml keys: {sorted(missing_keys)}"
        )

    return data


def normalize_class_names(
    names_value: list[str] | dict[int | str, str],
) -> dict[int, str]:
    """Convert YAML class names into an integer-keyed dictionary."""

    if isinstance(names_value, list):
        class_names = {
            index: str(name) for index, name in enumerate(names_value)
        }

    elif isinstance(names_value, dict):
        try:
            class_names = {
                int(class_id): str(name)
                for class_id, name in names_value.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All class-name dictionary keys must be integer-like."
            ) from exc

    else:
        raise ValueError(
            "'names' must be either a list or a class-ID dictionary."
        )

    expected_ids = list(range(len(class_names)))

    if sorted(class_names) != expected_ids:
        raise ValueError(
            "Class IDs in 'names' must be contiguous and start at zero. "
            f"Found IDs: {sorted(class_names)}"
        )

    return class_names


def resolve_dataset_root(yaml_path: Path, path_value: str) -> Path:
    """Resolve an absolute or data.yaml-relative dataset root."""

    dataset_root = Path(path_value).expanduser()

    if not dataset_root.is_absolute():
        dataset_root = yaml_path.parent / dataset_root

    return dataset_root.resolve()


def collect_images(image_directory: Path) -> list[Path]:
    """Collect supported image files recursively."""

    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_directory}"
        )

    return sorted(
        path
        for path in image_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def collect_labels(label_directory: Path) -> list[Path]:
    """Collect YOLO label text files recursively."""

    if not label_directory.exists():
        raise FileNotFoundError(
            f"Label directory does not exist: {label_directory}"
        )

    return sorted(
        path
        for path in label_directory.rglob("*.txt")
        if path.is_file()
    )


def relative_stem(path: Path, base_directory: Path) -> str:
    """
    Return a path-relative stem.

    This supports nested subdirectories by preserving the relative path while
    removing only the file extension.
    """

    relative_path = path.relative_to(base_directory)
    return relative_path.with_suffix("").as_posix()


def inspect_image(image_path: Path) -> tuple[bool, int | None, int | None, str]:
    """Verify an image and return validity, width, height, and error message."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            width, height = image.size

        if width <= 0 or height <= 0:
            return False, width, height, "Image has invalid dimensions."

        return True, width, height, ""

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as exc:
        return False, None, None, str(exc)


def inspect_label_file(
    label_path: Path,
    class_names: dict[int, str],
) -> tuple[
    Counter[int],
    list[str],
    list[str],
    int,
    int,
]:
    """
    Validate one YOLO label file.

    Returns:
        class_counts
        errors
        warnings
        valid_instance_count
        invalid_instance_count
    """

    class_counts: Counter[int] = Counter()
    errors: list[str] = []
    warnings: list[str] = []
    valid_instance_count = 0
    invalid_instance_count = 0

    try:
        content = label_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        errors.append(f"Could not read file: {exc}")
        return class_counts, errors, warnings, 0, 0

    if not content.strip():
        return class_counts, errors, warnings, 0, 0

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            warnings.append(f"Line {line_number}: blank line ignored.")
            continue

        values = line.split()

        if len(values) != 5:
            errors.append(
                f"Line {line_number}: expected 5 values, "
                f"found {len(values)}. Content: {line}"
            )
            invalid_instance_count += 1
            continue

        class_token, *coordinate_tokens = values

        try:
            class_float = float(class_token)

            if not class_float.is_integer():
                raise ValueError

            class_id = int(class_float)

        except ValueError:
            errors.append(
                f"Line {line_number}: class ID must be an integer. "
                f"Found: {class_token}"
            )
            invalid_instance_count += 1
            continue

        try:
            x_center, y_center, width, height = map(
                float,
                coordinate_tokens,
            )
        except ValueError:
            errors.append(
                f"Line {line_number}: coordinates must be numeric. "
                f"Content: {line}"
            )
            invalid_instance_count += 1
            continue

        coordinates = [x_center, y_center, width, height]

        if not all(math.isfinite(value) for value in coordinates):
            errors.append(
                f"Line {line_number}: coordinates contain NaN or infinity."
            )
            invalid_instance_count += 1
            continue

        line_errors: list[str] = []

        if class_id not in class_names:
            line_errors.append(
                f"invalid class ID {class_id}; expected "
                f"0 to {len(class_names) - 1}"
            )

        if not 0.0 <= x_center <= 1.0:
            line_errors.append(f"x_center={x_center} is outside [0, 1]")

        if not 0.0 <= y_center <= 1.0:
            line_errors.append(f"y_center={y_center} is outside [0, 1]")

        if not 0.0 < width <= 1.0:
            line_errors.append(f"width={width} must be in (0, 1]")

        if not 0.0 < height <= 1.0:
            line_errors.append(f"height={height} must be in (0, 1]")

        if line_errors:
            errors.append(
                f"Line {line_number}: " + "; ".join(line_errors)
            )
            invalid_instance_count += 1
            continue

        x_min = x_center - width / 2.0
        y_min = y_center - height / 2.0
        x_max = x_center + width / 2.0
        y_max = y_center + height / 2.0

        if x_min < 0.0 or y_min < 0.0 or x_max > 1.0 or y_max > 1.0:
            warnings.append(
                f"Line {line_number}: box extends beyond image boundary "
                f"(xmin={x_min:.6f}, ymin={y_min:.6f}, "
                f"xmax={x_max:.6f}, ymax={y_max:.6f})."
            )

        class_counts[class_id] += 1
        valid_instance_count += 1

    return (
        class_counts,
        errors,
        warnings,
        valid_instance_count,
        invalid_instance_count,
    )


def write_lines(path: Path, lines: list[str]) -> None:
    """Write report lines, including a message when no issues were found."""

    if lines:
        content = "\n".join(lines) + "\n"
    else:
        content = "No issues found.\n"

    path.write_text(content, encoding="utf-8")


def audit_split(
    split_name: str,
    image_directory: Path,
    label_directory: Path,
    class_names: dict[int, str],
) -> tuple[
    SplitStatistics,
    Counter[int],
    dict[str, list[str]],
]:
    """Audit one split and return statistics and issue lists."""

    statistics = SplitStatistics(split=split_name)
    class_counts: Counter[int] = Counter()

    issues: dict[str, list[str]] = {
        "missing_labels": [],
        "orphan_labels": [],
        "invalid_labels": [],
        "label_warnings": [],
        "empty_labels": [],
        "corrupted_images": [],
        "duplicate_image_stems": [],
    }

    images = collect_images(image_directory)
    labels = collect_labels(label_directory)

    statistics.image_count = len(images)
    statistics.label_file_count = len(labels)

    image_map: dict[str, Path] = {}
    duplicate_map: defaultdict[str, list[Path]] = defaultdict(list)

    for image_path in images:
        stem = relative_stem(image_path, image_directory)
        duplicate_map[stem].append(image_path)

        if stem not in image_map:
            image_map[stem] = image_path

    for stem, paths in duplicate_map.items():
        if len(paths) > 1:
            issues["duplicate_image_stems"].append(
                f"[{split_name}] {stem}: "
                + ", ".join(str(path) for path in paths)
            )

    label_map = {
        relative_stem(label_path, label_directory): label_path
        for label_path in labels
    }

    image_stems = set(image_map)
    label_stems = set(label_map)

    missing_label_stems = sorted(image_stems - label_stems)
    orphan_label_stems = sorted(label_stems - image_stems)
    matched_stems = sorted(image_stems & label_stems)

    statistics.missing_labels = len(missing_label_stems)
    statistics.orphan_labels = len(orphan_label_stems)
    statistics.matched_pairs = len(matched_stems)

    for stem in missing_label_stems:
        issues["missing_labels"].append(
            f"[{split_name}] {image_map[stem]}"
        )

    for stem in orphan_label_stems:
        issues["orphan_labels"].append(
            f"[{split_name}] {label_map[stem]}"
        )

    for image_path in images:
        valid, width, height, error = inspect_image(image_path)

        if not valid:
            statistics.corrupted_images += 1
            issues["corrupted_images"].append(
                f"[{split_name}] {image_path} | {error}"
            )

    for stem in matched_stems:
        label_path = label_map[stem]

        try:
            is_empty = not label_path.read_text(
                encoding="utf-8-sig"
            ).strip()
        except (OSError, UnicodeError):
            is_empty = False

        if is_empty:
            statistics.empty_labels += 1
            issues["empty_labels"].append(
                f"[{split_name}] {label_path}"
            )

        (
            file_class_counts,
            errors,
            warnings,
            valid_instances,
            invalid_instances,
        ) = inspect_label_file(label_path, class_names)

        class_counts.update(file_class_counts)
        statistics.valid_instances += valid_instances
        statistics.invalid_instances += invalid_instances

        if errors:
            statistics.invalid_label_files += 1

            for error in errors:
                issues["invalid_labels"].append(
                    f"[{split_name}] {label_path} | {error}"
                )

        for warning in warnings:
            issues["label_warnings"].append(
                f"[{split_name}] {label_path} | {warning}"
            )

    return statistics, class_counts, issues


def main() -> int:
    """Run the complete audit."""

    arguments = parse_arguments()

    yaml_path = arguments.data.expanduser().resolve()
    yaml_data = load_yaml(yaml_path)

    class_names = normalize_class_names(yaml_data["names"])

    declared_nc = yaml_data.get("nc")

    if declared_nc is not None and int(declared_nc) != len(class_names):
        raise ValueError(
            f"data.yaml declares nc={declared_nc}, but "
            f"{len(class_names)} class names were provided."
        )

    dataset_root = resolve_dataset_root(
        yaml_path=yaml_path,
        path_value=str(yaml_data["path"]),
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    output_directory = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else dataset_root / "audit_reports"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    split_definitions = {
        "train": yaml_data["train"],
        "val": yaml_data["val"],
    }

    all_split_statistics: list[SplitStatistics] = []
    total_class_counts: Counter[int] = Counter()
    per_split_class_counts: dict[str, Counter[int]] = {}
    combined_issues: defaultdict[str, list[str]] = defaultdict(list)

    print("=" * 72)
    print("YOLO DATASET AUDIT")
    print("=" * 72)
    print(f"data.yaml   : {yaml_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output      : {output_directory}")
    print(f"Classes     : {class_names}")
    print("=" * 72)

    for split_name, image_relative_path in split_definitions.items():
        image_directory = dataset_root / str(image_relative_path)

        try:
            relative_image_path = Path(str(image_relative_path))
            path_parts = list(relative_image_path.parts)

            if "images" not in path_parts:
                raise ValueError(
                    f"Split path does not contain an 'images' directory: "
                    f"{image_relative_path}"
                )

            images_index = path_parts.index("images")
            path_parts[images_index] = "labels"
            label_relative_path = Path(*path_parts)

        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not infer the label directory from "
                f"'{image_relative_path}'. Expected a path such as "
                f"'images/{split_name}'."
            ) from exc

        label_directory = dataset_root / label_relative_path

        print(f"\nAuditing split: {split_name}")
        print(f"Images: {image_directory}")
        print(f"Labels: {label_directory}")

        statistics, class_counts, issues = audit_split(
            split_name=split_name,
            image_directory=image_directory,
            label_directory=label_directory,
            class_names=class_names,
        )

        all_split_statistics.append(statistics)
        per_split_class_counts[split_name] = class_counts
        total_class_counts.update(class_counts)

        for issue_type, issue_lines in issues.items():
            combined_issues[issue_type].extend(issue_lines)

        print(f"  Images             : {statistics.image_count}")
        print(f"  Label files        : {statistics.label_file_count}")
        print(f"  Matched pairs      : {statistics.matched_pairs}")
        print(f"  Missing labels     : {statistics.missing_labels}")
        print(f"  Orphan labels      : {statistics.orphan_labels}")
        print(f"  Corrupted images   : {statistics.corrupted_images}")
        print(f"  Empty labels       : {statistics.empty_labels}")
        print(f"  Invalid label files: {statistics.invalid_label_files}")
        print(f"  Valid instances    : {statistics.valid_instances}")
        print(f"  Invalid instances  : {statistics.invalid_instances}")

    class_statistics_path = output_directory / "class_statistics.csv"

    with class_statistics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "class_id",
                "class_name",
                "train_instances",
                "val_instances",
                "total_instances",
            ]
        )

        for class_id, class_name in class_names.items():
            train_count = per_split_class_counts.get(
                "train",
                Counter(),
            )[class_id]

            val_count = per_split_class_counts.get(
                "val",
                Counter(),
            )[class_id]

            writer.writerow(
                [
                    class_id,
                    class_name,
                    train_count,
                    val_count,
                    train_count + val_count,
                ]
            )

    split_statistics_path = output_directory / "split_statistics.csv"

    with split_statistics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        field_names = list(asdict(all_split_statistics[0]).keys())
        writer = csv.DictWriter(file, fieldnames=field_names)

        writer.writeheader()

        for statistics in all_split_statistics:
            writer.writerow(asdict(statistics))

    report = {
        "data_yaml": str(yaml_path),
        "dataset_root": str(dataset_root),
        "class_count": len(class_names),
        "class_names": class_names,
        "splits": [
            asdict(statistics)
            for statistics in all_split_statistics
        ],
        "class_instances": {
            str(class_id): {
                "name": class_names[class_id],
                "train": per_split_class_counts.get(
                    "train",
                    Counter(),
                )[class_id],
                "val": per_split_class_counts.get(
                    "val",
                    Counter(),
                )[class_id],
                "total": total_class_counts[class_id],
            }
            for class_id in class_names
        },
        "issue_counts": {
            issue_type: len(lines)
            for issue_type, lines in combined_issues.items()
        },
    }

    report_path = output_directory / "dataset_audit_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    issue_file_mapping = {
        "missing_labels": "missing_labels.txt",
        "orphan_labels": "orphan_labels.txt",
        "invalid_labels": "invalid_labels.txt",
        "label_warnings": "label_warnings.txt",
        "empty_labels": "empty_labels.txt",
        "corrupted_images": "corrupted_images.txt",
        "duplicate_image_stems": "duplicate_image_stems.txt",
    }

    for issue_type, output_filename in issue_file_mapping.items():
        write_lines(
            output_directory / output_filename,
            combined_issues[issue_type],
        )

    fatal_issue_count = sum(
        [
            len(combined_issues["missing_labels"]),
            len(combined_issues["orphan_labels"]),
            len(combined_issues["invalid_labels"]),
            len(combined_issues["corrupted_images"]),
            len(combined_issues["duplicate_image_stems"]),
        ]
    )

    print("\n" + "=" * 72)
    print("CLASS COUNTS")
    print("=" * 72)

    for class_id, class_name in class_names.items():
        print(
            f"{class_id}: {class_name:<15} "
            f"train={per_split_class_counts['train'][class_id]:>7} "
            f"val={per_split_class_counts['val'][class_id]:>7} "
            f"total={total_class_counts[class_id]:>7}"
        )

    print("\n" + "=" * 72)
    print("AUDIT COMPLETE")
    print("=" * 72)
    print(f"Reports saved to: {output_directory}")
    print(f"Fatal issue count: {fatal_issue_count}")
    print(
        f"Boundary warnings: "
        f"{len(combined_issues['label_warnings'])}"
    )

    if fatal_issue_count == 0:
        print("STATUS: Dataset passed the structural audit.")
        return 0

    print("STATUS: Dataset contains issues that must be reviewed.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("\nDATASET AUDIT FAILED")
        print(f"{type(error).__name__}: {error}")
        sys.exit(2)