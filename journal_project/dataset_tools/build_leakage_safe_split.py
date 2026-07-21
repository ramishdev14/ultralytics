from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class FrameRecord:
    image_path: Path
    label_path: Path
    original_split: str
    sequence_id: str
    frame_number: int
    filename: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a weather-stratified, temporally separated YOLO "
            "train/validation/test dataset from extracted video frames."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset"),
        help=(
            "Original YOLO dataset containing images/train, images/val, "
            "labels/train and labels/val."
        ),
    )

    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("dataset_grouped_final"),
        help="Output directory for the new dataset.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training proportion within every sequence.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation proportion within every sequence.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Testing proportion within every sequence.",
    )

    parser.add_argument(
        "--gap-per-side",
        type=int,
        default=3,
        help=(
            "Frames removed from each side of every cross-split boundary. "
            "A value of 3 removes 6 frames per boundary."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to randomize temporal split-order rotation.",
    )

    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Whether to copy files or create hard links.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination first if it already exists.",
    )

    parser.add_argument(
        "--class-names",
        nargs="+",
        default=[
            "car",
            "truck",
            "bus",
            "traffic_sign",
        ],
        help="YOLO class names in class-index order.",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    ratios = (
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero.")

    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(
            "train-ratio + val-ratio + test-ratio must equal 1.0. "
            f"Received {sum(ratios):.8f}."
        )

    if args.gap_per_side < 0:
        raise ValueError("gap-per-side cannot be negative.")

    source = args.source.resolve()
    destination = args.destination.resolve()

    if source == destination:
        raise ValueError(
            "Source and destination directories must be different."
        )

    required_directories = (
        source / "images" / "train",
        source / "images" / "val",
        source / "labels" / "train",
        source / "labels" / "val",
    )

    for directory in required_directories:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required directory was not found: {directory}"
            )


def natural_sort_key(text: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def parse_sequence_and_frame(stem: str) -> tuple[str, int]:
    """
    Extract the sequence identifier and final frame number.

    Examples
    --------
    cam_1_dawn104 -> sequence: cam_1_dawn, frame: 104
    cam_1_rain_22 -> sequence: cam_1_rain, frame: 22
    cam_9_158     -> sequence: cam_9, frame: 158
    """

    match = re.match(r"^(.*?)(?:[_-]?)(\d+)$", stem)

    if match is None:
        raise ValueError(
            "Could not extract a trailing frame number from filename: "
            f"{stem}"
        )

    sequence_id = match.group(1).rstrip("_- ")
    frame_number = int(match.group(2))

    if not sequence_id:
        raise ValueError(
            f"Could not infer a sequence identifier from: {stem}"
        )

    return sequence_id, frame_number


def discover_images(directory: Path) -> list[Path]:
    images = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(
        images,
        key=lambda path: natural_sort_key(path.name),
    )


def read_label_counts(
    label_path: Path,
    number_of_classes: int,
) -> Counter:
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing annotation file: {label_path}"
        )

    text = label_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    counts: Counter = Counter()

    if not text:
        return counts

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.split()

        if len(parts) < 5:
            raise ValueError(
                f"Invalid YOLO annotation in {label_path}, "
                f"line {line_number}: {line!r}"
            )

        try:
            class_id = int(float(parts[0]))
        except ValueError as error:
            raise ValueError(
                f"Invalid class identifier in {label_path}, "
                f"line {line_number}: {parts[0]!r}"
            ) from error

        if not 0 <= class_id < number_of_classes:
            raise ValueError(
                f"Class ID {class_id} in {label_path} is outside "
                f"the valid range 0–{number_of_classes - 1}."
            )

        counts[class_id] += 1

    return counts


def collect_records(
    source: Path,
    class_names: list[str],
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    filenames_seen: dict[str, Path] = {}

    for original_split in ("train", "val"):
        image_directory = source / "images" / original_split
        label_directory = source / "labels" / original_split

        images = discover_images(image_directory)

        print(
            f"Found {len(images)} images in "
            f"{image_directory}"
        )

        for image_path in images:
            previous_path = filenames_seen.get(image_path.name)

            if previous_path is not None:
                raise ValueError(
                    "Duplicate filename detected across the original "
                    "dataset folders:\n"
                    f"  {previous_path}\n"
                    f"  {image_path}"
                )

            filenames_seen[image_path.name] = image_path

            sequence_id, frame_number = parse_sequence_and_frame(
                image_path.stem
            )

            label_path = (
                label_directory / f"{image_path.stem}.txt"
            )

            # Validate the annotation while collecting records.
            read_label_counts(
                label_path=label_path,
                number_of_classes=len(class_names),
            )

            records.append(
                FrameRecord(
                    image_path=image_path,
                    label_path=label_path,
                    original_split=original_split,
                    sequence_id=sequence_id,
                    frame_number=frame_number,
                    filename=image_path.name,
                )
            )

    records.sort(
        key=lambda record: (
            natural_sort_key(record.sequence_id),
            record.frame_number,
            natural_sort_key(record.filename),
        )
    )

    return records


def verify_unique_sequence_frames(
    records: list[FrameRecord],
) -> None:
    seen: dict[tuple[str, int], FrameRecord] = {}

    for record in records:
        key = (
            record.sequence_id,
            record.frame_number,
        )

        previous = seen.get(key)

        if previous is not None:
            raise ValueError(
                "Two files were parsed as the same sequence and frame:\n"
                f"  First:  {previous.image_path}\n"
                f"  Second: {record.image_path}\n"
                f"  Parsed sequence/frame: {key}"
            )

        seen[key] = record


def group_records_by_sequence(
    records: list[FrameRecord],
) -> dict[str, list[FrameRecord]]:
    grouped: dict[str, list[FrameRecord]] = defaultdict(list)

    for record in records:
        grouped[record.sequence_id].append(record)

    for sequence_id in grouped:
        grouped[sequence_id].sort(
            key=lambda record: (
                record.frame_number,
                natural_sort_key(record.filename),
            )
        )

    return dict(grouped)


def allocate_split_counts(
    available_frames: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    """
    Allocate an exact number of retained frames using the largest-remainder
    method while ensuring that every split receives at least one frame.
    """

    if available_frames < 3:
        raise ValueError(
            "At least three retained frames are required to represent "
            "train, validation and test."
        )

    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    raw_counts = {
        split_name: available_frames * ratio
        for split_name, ratio in ratios.items()
    }

    counts = {
        split_name: int(raw_count)
        for split_name, raw_count in raw_counts.items()
    }

    # Make sure every split has at least one frame.
    for split_name in SPLIT_NAMES:
        if counts[split_name] == 0:
            counts[split_name] = 1

    current_total = sum(counts.values())

    if current_total > available_frames:
        removable = sorted(
            SPLIT_NAMES,
            key=lambda name: counts[name],
            reverse=True,
        )

        while current_total > available_frames:
            changed = False

            for split_name in removable:
                if counts[split_name] > 1:
                    counts[split_name] -= 1
                    current_total -= 1
                    changed = True

                    if current_total == available_frames:
                        break

            if not changed:
                raise RuntimeError(
                    "Could not allocate at least one frame to every split."
                )

    remaining = available_frames - current_total

    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda split_name: (
            raw_counts[split_name] - int(raw_counts[split_name])
        ),
        reverse=True,
    )

    index = 0

    while remaining > 0:
        split_name = remainder_order[
            index % len(remainder_order)
        ]

        counts[split_name] += 1
        remaining -= 1
        index += 1

    if sum(counts.values()) != available_frames:
        raise RuntimeError(
            "Internal allocation error: counts do not match "
            "the available frame total."
        )

    return counts


def get_rotated_temporal_order(
    rotation_index: int,
) -> tuple[str, str, str]:
    """
    Rotate which split occupies the beginning, middle and end of a sequence.

    Rotation 0:
        train -> val -> test

    Rotation 1:
        val -> test -> train

    Rotation 2:
        test -> train -> val
    """

    orders = (
        ("train", "val", "test"),
        ("val", "test", "train"),
        ("test", "train", "val"),
    )

    return orders[rotation_index % len(orders)]


def split_sequences(
    grouped_records: dict[str, list[FrameRecord]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    gap_per_side: int,
    seed: int,
) -> tuple[
    dict[str, list[FrameRecord]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """
    Divide every sequence into three large contiguous split blocks.

    Between block 1 and block 2:
        gap_per_side frames are removed from the right side of block 1,
        and gap_per_side frames are removed from the left side of block 2.

    The same applies between block 2 and block 3.

    Therefore, each sequence discards at most:

        4 * gap_per_side

    frames.
    """

    records_by_split: dict[str, list[FrameRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    discarded_records: list[dict[str, object]] = []
    sequence_manifest: list[dict[str, object]] = []

    sequence_ids = sorted(
        grouped_records,
        key=natural_sort_key,
    )

    # Randomize which sequence receives which temporal rotation while
    # keeping the output reproducible.
    rng = random.Random(seed)
    rotation_sequence_ids = list(sequence_ids)
    rng.shuffle(rotation_sequence_ids)

    rotation_by_sequence = {
        sequence_id: index % 3
        for index, sequence_id
        in enumerate(rotation_sequence_ids)
    }

    total_gap_frames_per_sequence = 4 * gap_per_side

    for sequence_id in sequence_ids:
        sequence_records = grouped_records[sequence_id]
        sequence_length = len(sequence_records)

        retained_length = (
            sequence_length - total_gap_frames_per_sequence
        )

        if retained_length < 3:
            raise ValueError(
                f"Sequence {sequence_id!r} contains only "
                f"{sequence_length} frames. After removing "
                f"{total_gap_frames_per_sequence} boundary frames, "
                "there are not enough frames for all three splits."
            )

        split_counts = allocate_split_counts(
            available_frames=retained_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        rotation_index = rotation_by_sequence[
            sequence_id
        ]

        temporal_order = get_rotated_temporal_order(
            rotation_index
        )

        cursor = 0
        assigned_ranges: dict[str, dict[str, int]] = {}

        for block_index, split_name in enumerate(
            temporal_order
        ):
            block_size = split_counts[split_name]

            block_records = sequence_records[
                cursor : cursor + block_size
            ]

            if len(block_records) != block_size:
                raise RuntimeError(
                    f"Internal block-size error for sequence "
                    f"{sequence_id}, split {split_name}."
                )

            records_by_split[split_name].extend(
                block_records
            )

            assigned_ranges[split_name] = {
                "first_frame": (
                    block_records[0].frame_number
                ),
                "last_frame": (
                    block_records[-1].frame_number
                ),
                "images": len(block_records),
            }

            cursor += block_size

            # Insert one temporal exclusion region between neighboring
            # split blocks.
            if block_index < 2 and gap_per_side > 0:
                gap_size = 2 * gap_per_side

                gap_records = sequence_records[
                    cursor : cursor + gap_size
                ]

                if len(gap_records) != gap_size:
                    raise RuntimeError(
                        f"Could not create the requested temporal gap "
                        f"for sequence {sequence_id}."
                    )

                left_split = temporal_order[block_index]
                right_split = temporal_order[
                    block_index + 1
                ]

                for gap_record in gap_records:
                    discarded_records.append(
                        {
                            "filename": gap_record.filename,
                            "sequence_id": (
                                gap_record.sequence_id
                            ),
                            "frame_number": (
                                gap_record.frame_number
                            ),
                            "original_split": (
                                gap_record.original_split
                            ),
                            "reason": (
                                "temporal_boundary_gap_between_"
                                f"{left_split}_and_{right_split}"
                            ),
                        }
                    )

                cursor += gap_size

        if cursor != sequence_length:
            raise RuntimeError(
                f"Sequence allocation did not consume all frames for "
                f"{sequence_id}. Used {cursor}, available "
                f"{sequence_length}."
            )

        sequence_manifest.append(
            {
                "sequence_id": sequence_id,
                "original_images": sequence_length,
                "retained_images": retained_length,
                "discarded_images": (
                    sequence_length - retained_length
                ),
                "rotation_index": rotation_index,
                "temporal_order": " -> ".join(
                    temporal_order
                ),
                "train_images": (
                    assigned_ranges["train"]["images"]
                ),
                "train_first_frame": (
                    assigned_ranges["train"]["first_frame"]
                ),
                "train_last_frame": (
                    assigned_ranges["train"]["last_frame"]
                ),
                "val_images": (
                    assigned_ranges["val"]["images"]
                ),
                "val_first_frame": (
                    assigned_ranges["val"]["first_frame"]
                ),
                "val_last_frame": (
                    assigned_ranges["val"]["last_frame"]
                ),
                "test_images": (
                    assigned_ranges["test"]["images"]
                ),
                "test_first_frame": (
                    assigned_ranges["test"]["first_frame"]
                ),
                "test_last_frame": (
                    assigned_ranges["test"]["last_frame"]
                ),
            }
        )

    for split_name in SPLIT_NAMES:
        records_by_split[split_name].sort(
            key=lambda record: (
                natural_sort_key(record.sequence_id),
                record.frame_number,
            )
        )

    return (
        records_by_split,
        discarded_records,
        sequence_manifest,
    )


def verify_no_filename_overlap(
    records_by_split: dict[str, list[FrameRecord]],
) -> None:
    filenames = {
        split_name: {
            record.filename
            for record in records
        }
        for split_name, records
        in records_by_split.items()
    }

    pairs = (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    )

    for left_split, right_split in pairs:
        overlap = (
            filenames[left_split]
            & filenames[right_split]
        )

        if overlap:
            raise RuntimeError(
                f"Filename overlap detected between {left_split} "
                f"and {right_split}: {sorted(overlap)[:10]}"
            )


def calculate_minimum_temporal_distances(
    records_by_split: dict[str, list[FrameRecord]],
) -> list[dict[str, object]]:
    frames_by_sequence_and_split: dict[
        str,
        dict[str, list[int]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for split_name, records in records_by_split.items():
        for record in records:
            frames_by_sequence_and_split[
                record.sequence_id
            ][split_name].append(
                record.frame_number
            )

    results: list[dict[str, object]] = []

    split_pairs = (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    )

    for sequence_id, split_frames in (
        frames_by_sequence_and_split.items()
    ):
        for left_split, right_split in split_pairs:
            left_frames = sorted(
                split_frames.get(left_split, [])
            )

            right_frames = sorted(
                split_frames.get(right_split, [])
            )

            if not left_frames or not right_frames:
                continue

            left_index = 0
            right_index = 0

            minimum_distance: int | None = None
            closest_pair: tuple[int, int] | None = None

            while (
                left_index < len(left_frames)
                and right_index < len(right_frames)
            ):
                left_frame = left_frames[left_index]
                right_frame = right_frames[right_index]

                distance = abs(
                    left_frame - right_frame
                )

                if (
                    minimum_distance is None
                    or distance < minimum_distance
                ):
                    minimum_distance = distance
                    closest_pair = (
                        left_frame,
                        right_frame,
                    )

                if left_frame < right_frame:
                    left_index += 1
                else:
                    right_index += 1

            results.append(
                {
                    "sequence_id": sequence_id,
                    "left_split": left_split,
                    "right_split": right_split,
                    "minimum_frame_number_distance": (
                        minimum_distance
                    ),
                    "closest_left_frame": (
                        closest_pair[0]
                        if closest_pair
                        else None
                    ),
                    "closest_right_frame": (
                        closest_pair[1]
                        if closest_pair
                        else None
                    ),
                }
            )

    return results


def verify_temporal_separation(
    temporal_distances: list[dict[str, object]],
    gap_per_side: int,
) -> None:
    """
    With a gap of g removed from each side, neighboring retained frames
    should differ by at least:

        2g + 1

    in their frame numbers, assuming consecutive numbering.
    """

    expected_minimum_distance = (
        2 * gap_per_side + 1
    )

    violations = [
        row
        for row in temporal_distances
        if row["minimum_frame_number_distance"]
        is not None
        and int(
            row["minimum_frame_number_distance"]
        ) < expected_minimum_distance
    ]

    if violations:
        examples = violations[:10]

        raise RuntimeError(
            "Temporal separation verification failed. "
            f"Expected a minimum frame-number distance of "
            f"{expected_minimum_distance}.\n"
            f"Examples: {examples}"
        )


def prepare_destination(
    destination: Path,
    overwrite: bool,
) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}\n"
                "Use --overwrite to rebuild it."
            )

        print(
            f"Removing existing destination: "
            f"{destination}"
        )

        shutil.rmtree(destination)

    for split_name in SPLIT_NAMES:
        (
            destination
            / "images"
            / split_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            destination
            / "labels"
            / split_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    (
        destination
        / "split_manifests"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def transfer_file(
    source: Path,
    destination: Path,
    copy_mode: str,
) -> None:
    if copy_mode == "copy":
        shutil.copy2(
            source,
            destination,
        )
        return

    if copy_mode == "hardlink":
        try:
            destination.hardlink_to(
                source.resolve()
            )
        except OSError as error:
            raise OSError(
                f"Could not create hard link:\n"
                f"  Source:      {source}\n"
                f"  Destination: {destination}\n"
                "Use --copy-mode copy if the folders are "
                "on different drives."
            ) from error

        return

    raise ValueError(
        f"Unsupported copy mode: {copy_mode}"
    )


def materialize_dataset(
    records_by_split: dict[str, list[FrameRecord]],
    destination: Path,
    copy_mode: str,
) -> None:
    total_records = sum(
        len(records)
        for records in records_by_split.values()
    )

    transferred = 0

    for split_name in SPLIT_NAMES:
        image_destination = (
            destination
            / "images"
            / split_name
        )

        label_destination = (
            destination
            / "labels"
            / split_name
        )

        for record in records_by_split[split_name]:
            transfer_file(
                source=record.image_path,
                destination=(
                    image_destination
                    / record.image_path.name
                ),
                copy_mode=copy_mode,
            )

            transfer_file(
                source=record.label_path,
                destination=(
                    label_destination
                    / record.label_path.name
                ),
                copy_mode=copy_mode,
            )

            transferred += 1

            if (
                transferred % 1000 == 0
                or transferred == total_records
            ):
                print(
                    f"Transferred {transferred}/"
                    f"{total_records} samples..."
                )


def verify_output_pairs(
    destination: Path,
) -> None:
    for split_name in SPLIT_NAMES:
        image_directory = (
            destination
            / "images"
            / split_name
        )

        label_directory = (
            destination
            / "labels"
            / split_name
        )

        image_stems = {
            image_path.stem
            for image_path
            in discover_images(image_directory)
        }

        label_stems = {
            label_path.stem
            for label_path
            in label_directory.glob("*.txt")
        }

        missing_labels = (
            image_stems - label_stems
        )

        orphan_labels = (
            label_stems - image_stems
        )

        if missing_labels:
            raise RuntimeError(
                f"{split_name} contains images without labels: "
                f"{sorted(missing_labels)[:10]}"
            )

        if orphan_labels:
            raise RuntimeError(
                f"{split_name} contains labels without images: "
                f"{sorted(orphan_labels)[:10]}"
            )


def calculate_statistics(
    records_by_split: dict[str, list[FrameRecord]],
    class_names: list[str],
) -> dict[str, object]:
    total_images = sum(
        len(records)
        for records in records_by_split.values()
    )

    report: dict[str, object] = {
        "splits": {},
    }

    total_class_counts: Counter = Counter()
    total_empty_labels = 0

    for split_name in SPLIT_NAMES:
        records = records_by_split[split_name]

        class_counts: Counter = Counter()
        empty_labels = 0
        sequences: set[str] = set()

        for record in records:
            counts = read_label_counts(
                label_path=record.label_path,
                number_of_classes=len(class_names),
            )

            class_counts.update(counts)

            sequences.add(record.sequence_id)

            if not counts:
                empty_labels += 1

        total_class_counts.update(class_counts)
        total_empty_labels += empty_labels

        report["splits"][split_name] = {
            "images": len(records),
            "percentage": (
                100.0
                * len(records)
                / max(total_images, 1)
            ),
            "sequences_represented": len(
                sequences
            ),
            "empty_labels": empty_labels,
            "objects": sum(
                class_counts.values()
            ),
            "class_counts": {
                class_names[class_id]: (
                    class_counts[class_id]
                )
                for class_id
                in range(len(class_names))
            },
        }

    report["total_images"] = total_images
    report["total_empty_labels"] = (
        total_empty_labels
    )
    report["total_objects"] = sum(
        total_class_counts.values()
    )

    report["total_class_counts"] = {
        class_names[class_id]: (
            total_class_counts[class_id]
        )
        for class_id
        in range(len(class_names))
    }

    return report


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_manifests(
    records_by_split: dict[str, list[FrameRecord]],
    discarded_records: list[dict[str, object]],
    sequence_manifest: list[dict[str, object]],
    temporal_distances: list[dict[str, object]],
    destination: Path,
) -> None:
    manifest_directory = (
        destination / "split_manifests"
    )

    sample_rows: list[dict[str, object]] = []

    for split_name in SPLIT_NAMES:
        for record in records_by_split[split_name]:
            sample_rows.append(
                {
                    "filename": record.filename,
                    "label_filename": (
                        record.label_path.name
                    ),
                    "sequence_id": (
                        record.sequence_id
                    ),
                    "frame_number": (
                        record.frame_number
                    ),
                    "new_split": split_name,
                    "original_split": (
                        record.original_split
                    ),
                    "source_image": str(
                        record.image_path
                    ),
                    "source_label": str(
                        record.label_path
                    ),
                }
            )

    write_csv(
        path=(
            manifest_directory
            / "sample_assignments.csv"
        ),
        fieldnames=[
            "filename",
            "label_filename",
            "sequence_id",
            "frame_number",
            "new_split",
            "original_split",
            "source_image",
            "source_label",
        ],
        rows=sample_rows,
    )

    write_csv(
        path=(
            manifest_directory
            / "discarded_boundary_frames.csv"
        ),
        fieldnames=[
            "filename",
            "sequence_id",
            "frame_number",
            "original_split",
            "reason",
        ],
        rows=discarded_records,
    )

    write_csv(
        path=(
            manifest_directory
            / "sequence_split_summary.csv"
        ),
        fieldnames=[
            "sequence_id",
            "original_images",
            "retained_images",
            "discarded_images",
            "rotation_index",
            "temporal_order",
            "train_images",
            "train_first_frame",
            "train_last_frame",
            "val_images",
            "val_first_frame",
            "val_last_frame",
            "test_images",
            "test_first_frame",
            "test_last_frame",
        ],
        rows=sequence_manifest,
    )

    write_csv(
        path=(
            manifest_directory
            / "temporal_separation.csv"
        ),
        fieldnames=[
            "sequence_id",
            "left_split",
            "right_split",
            "minimum_frame_number_distance",
            "closest_left_frame",
            "closest_right_frame",
        ],
        rows=temporal_distances,
    )


def write_dataset_yaml(
    destination: Path,
    class_names: list[str],
) -> None:
    yaml_lines = [
        f"path: {destination.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]

    for class_id, class_name in enumerate(
        class_names
    ):
        yaml_lines.append(
            f"  {class_id}: {class_name}"
        )

    (
        destination / "data.yaml"
    ).write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def print_statistics(
    report: dict[str, object],
) -> None:
    print("\n" + "=" * 78)
    print("FINAL DATASET STATISTICS")
    print("=" * 78)

    for split_name in SPLIT_NAMES:
        split_report = report["splits"][
            split_name
        ]

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"  Images:       "
            f"{split_report['images']} "
            f"({split_report['percentage']:.2f}%)"
        )

        print(
            f"  Objects:      "
            f"{split_report['objects']}"
        )

        print(
            f"  Empty labels: "
            f"{split_report['empty_labels']}"
        )

        print(
            f"  Sequences:    "
            f"{split_report['sequences_represented']}"
        )

        print("  Class counts:")

        for class_name, count in (
            split_report["class_counts"].items()
        ):
            print(
                f"    {class_name:15s}: "
                f"{count}"
            )

    print("\nTOTAL")

    print(
        f"  Images:       "
        f"{report['total_images']}"
    )

    print(
        f"  Objects:      "
        f"{report['total_objects']}"
    )

    print(
        f"  Empty labels: "
        f"{report['total_empty_labels']}"
    )

    print("=" * 78)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    source = args.source.resolve()
    destination = args.destination.resolve()

    print("=" * 78)
    print("WEATHER-STRATIFIED TEMPORAL DATASET SPLITTER")
    print("=" * 78)

    print(f"Source:             {source}")
    print(f"Destination:        {destination}")
    print(f"Train ratio:        {args.train_ratio}")
    print(f"Validation ratio:   {args.val_ratio}")
    print(f"Test ratio:         {args.test_ratio}")
    print(f"Gap per side:       {args.gap_per_side}")
    print(
        f"Frames per boundary:"
        f" {2 * args.gap_per_side}"
    )
    print(f"Random seed:        {args.seed}")
    print(f"Copy mode:          {args.copy_mode}")

    records = collect_records(
        source=source,
        class_names=args.class_names,
    )

    print(
        f"\nCollected {len(records)} total images."
    )

    verify_unique_sequence_frames(records)

    grouped_records = group_records_by_sequence(
        records
    )

    print(
        f"Detected {len(grouped_records)} sequences."
    )

    print("\nSequence sizes:")

    for sequence_id in sorted(
        grouped_records,
        key=natural_sort_key,
    ):
        print(
            f"  {sequence_id:30s}: "
            f"{len(grouped_records[sequence_id]):5d} frames"
        )

    (
        records_by_split,
        discarded_records,
        sequence_manifest,
    ) = split_sequences(
        grouped_records=grouped_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        gap_per_side=args.gap_per_side,
        seed=args.seed,
    )

    verify_no_filename_overlap(
        records_by_split
    )

    temporal_distances = (
        calculate_minimum_temporal_distances(
            records_by_split
        )
    )

    verify_temporal_separation(
        temporal_distances=temporal_distances,
        gap_per_side=args.gap_per_side,
    )

    print(
        f"\nDiscarded {len(discarded_records)} "
        "temporal boundary frames."
    )

    prepare_destination(
        destination=destination,
        overwrite=args.overwrite,
    )

    materialize_dataset(
        records_by_split=records_by_split,
        destination=destination,
        copy_mode=args.copy_mode,
    )

    verify_output_pairs(destination)

    write_manifests(
        records_by_split=records_by_split,
        discarded_records=discarded_records,
        sequence_manifest=sequence_manifest,
        temporal_distances=temporal_distances,
        destination=destination,
    )

    report = calculate_statistics(
        records_by_split=records_by_split,
        class_names=args.class_names,
    )

    report["configuration"] = {
        "source": str(source),
        "destination": str(destination),
        "method": (
            "weather_stratified_contiguous_blocks"
        ),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "gap_per_side": args.gap_per_side,
        "frames_removed_per_boundary": (
            2 * args.gap_per_side
        ),
        "maximum_removed_per_sequence": (
            4 * args.gap_per_side
        ),
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "class_names": args.class_names,
        "detected_sequences": len(
            grouped_records
        ),
        "discarded_boundary_frames": len(
            discarded_records
        ),
        "temporal_separation_verified": True,
        "minimum_expected_frame_distance": (
            2 * args.gap_per_side + 1
        ),
    }

    report_path = (
        destination
        / "split_manifests"
        / "split_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_dataset_yaml(
        destination=destination,
        class_names=args.class_names,
    )

    print_statistics(report)

    print(
        "\nTemporal separation verification: PASSED"
    )

    print("\nCreated:")
    print(
        f"  {destination / 'data.yaml'}"
    )
    print(
        f"  {destination / 'split_manifests' / 'split_report.json'}"
    )
    print(
        f"  {destination / 'split_manifests' / 'sequence_split_summary.csv'}"
    )
    print(
        f"  {destination / 'split_manifests' / 'sample_assignments.csv'}"
    )
    print(
        f"  {destination / 'split_manifests' / 'discarded_boundary_frames.csv'}"
    )
    print(
        f"  {destination / 'split_manifests' / 'temporal_separation.csv'}"
    )

    print(
        "\nDataset creation completed successfully."
    )


if __name__ == "__main__":
    main()