"""
src/training/dataset_splitter.py
Dataset dogrulama, train/valid/test ayirma ve data.yaml uretimi.
"""
from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_DIR = Path(config.get("app", "data_dirs", "dataset", default="data/dataset"))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    label_path: Path


def _iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()])


def _validate_label_file(label_path: Path, max_class_id: int) -> bool:
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    return False
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id > max_class_id:
                    return False
                cx, cy, w, h = [float(v) for v in parts[1:]]
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    return False
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    return False
        return True
    except Exception:
        return False


def _validate_image_file(image_path: Path) -> bool:
    try:
        image = cv2.imread(str(image_path))
        return image is not None and image.size > 0
    except Exception:
        return False


def _discover_source_roots(dataset_path: Path) -> list[tuple[Path, Path]]:
    """
    Eski ve yeni layout'lari destekler:
      - images/train + labels/train
      - images/val + labels/val
      - images/test + labels/test
      - images + labels
    """
    candidates = [
        (dataset_path / "images" / "train", dataset_path / "labels" / "train"),
        (dataset_path / "images" / "val", dataset_path / "labels" / "val"),
        (dataset_path / "images" / "test", dataset_path / "labels" / "test"),
        (dataset_path / "images", dataset_path / "labels"),
    ]
    roots: list[tuple[Path, Path]] = []
    for images_dir, labels_dir in candidates:
        if images_dir.exists() and labels_dir.exists():
            roots.append((images_dir, labels_dir))
    return roots


def _collect_pairs(dataset_path: Path, class_names: list[str]) -> tuple[list[SamplePair], dict]:
    roots = _discover_source_roots(dataset_path)
    if not roots:
        raise RuntimeError(f"No dataset source roots found under: {dataset_path}")

    max_class_id = max(len(class_names) - 1, 0)
    pairs: list[SamplePair] = []
    report = {
        "sources": [str(img_dir) for img_dir, _ in roots],
        "total_images_scanned": 0,
        "valid_pairs": 0,
        "missing_labels": 0,
        "corrupt_images": 0,
        "invalid_labels": 0,
        "orphan_labels": 0,
    }

    seen_stems: set[str] = set()

    for images_dir, labels_dir in roots:
        images = _iter_images(images_dir)
        report["total_images_scanned"] += len(images)

        label_stems = {p.stem for p in labels_dir.glob("*.txt") if p.is_file()}
        image_stems = {p.stem for p in images}

        orphan_count = len(label_stems - image_stems)
        report["orphan_labels"] += orphan_count

        for image_path in images:
            stem = image_path.stem
            if stem in seen_stems:
                continue

            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                report["missing_labels"] += 1
                continue

            if not _validate_image_file(image_path):
                report["corrupt_images"] += 1
                continue

            if not _validate_label_file(label_path, max_class_id):
                report["invalid_labels"] += 1
                continue

            pairs.append(SamplePair(image_path=image_path, label_path=label_path))
            seen_stems.add(stem)

    report["valid_pairs"] = len(pairs)
    return pairs, report


def _split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_valid_samples: int = 1,
    min_test_samples: int = 1,
) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    targets = {
        "train": total * train_ratio,
        "valid": total * val_ratio,
        "test": total * test_ratio,
    }
    counts = {name: int(value) for name, value in targets.items()}

    # Kalan ornekleri en buyuk kesirli kisma dagit.
    remainder = total - sum(counts.values())
    if remainder > 0:
        ordered = sorted(
            targets.keys(),
            key=lambda k: (targets[k] - counts[k]),
            reverse=True,
        )
        idx = 0
        while remainder > 0:
            counts[ordered[idx % len(ordered)]] += 1
            remainder -= 1
            idx += 1

    # Kucuk veri setlerinde val/test'in bos kalmasi egitimi kirar.
    # total >= 3 ise train/valid/test en az 1 ornek alsin.
    if total >= 3:
        for split in ["valid", "test"]:
            min_required = min_valid_samples if split == "valid" else min_test_samples
            min_required = max(min_required, 1)
            if counts[split] == 0:
                # Once train'den al, train 1'in altina dusmesin.
                if counts["train"] > 1:
                    counts["train"] -= 1
                    counts[split] += 1
                # Gerekirse diger split'ten al.
                elif split == "valid" and counts["test"] > 1:
                    counts["test"] -= 1
                    counts["valid"] += 1
                elif split == "test" and counts["valid"] > 1:
                    counts["valid"] -= 1
                    counts["test"] += 1

        if counts["train"] == 0:
            donor = "valid" if counts["valid"] > counts["test"] else "test"
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["train"] += 1

    # Test/valid stabilitesi icin minimum ornek sayilarini zorla.
    # Donor seciminde train her zaman en az 1 kalir.
    for split, min_required in (("valid", max(min_valid_samples, 1)), ("test", max(min_test_samples, 1))):
        while total >= 3 and counts[split] < min_required:
            donor_candidates = [
                ("train", 1),
                ("valid", max(min_valid_samples, 1) if split != "valid" else 0),
                ("test", max(min_test_samples, 1) if split != "test" else 0),
            ]
            donor = None
            donor_count = -1
            for donor_name, donor_floor in donor_candidates:
                if donor_name == split:
                    continue
                if counts[donor_name] > donor_floor and counts[donor_name] > donor_count:
                    donor = donor_name
                    donor_count = counts[donor_name]

            if donor is None:
                break

            counts[donor] -= 1
            counts[split] += 1

    if sum(counts.values()) != total:
        raise ValueError("Invalid split counts: total mismatch")

    return counts["train"], counts["valid"], counts["test"]


def _prepare_output_dirs(dataset_path: Path, clear_output_dirs: bool) -> dict[str, tuple[Path, Path]]:
    mapping = {
        "train": (dataset_path / "train" / "images", dataset_path / "train" / "labels"),
        "valid": (dataset_path / "valid" / "images", dataset_path / "valid" / "labels"),
        "test": (dataset_path / "test" / "images", dataset_path / "test" / "labels"),
    }

    if clear_output_dirs:
        for split in ["train", "valid", "test"]:
            split_dir = dataset_path / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    for images_dir, labels_dir in mapping.values():
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    return mapping


def _copy_pairs(pairs: list[SamplePair], target_images_dir: Path, target_labels_dir: Path) -> None:
    for pair in pairs:
        shutil.copy2(pair.image_path, target_images_dir / pair.image_path.name)
        shutil.copy2(pair.label_path, target_labels_dir / pair.label_path.name)


def _sanitize_label_file_inplace(label_path: Path) -> dict:
    """
    Label dosyasinda asagidaki kalite sorunlarini temizler:
      - birebir tekrar satirlar
      - ayni bbox koordinatina birden fazla class atanmasi (conflict)
    Conflict durumunda en sik gorulen class secilir (esitlikte en kucuk class id).
    """
    stats = {
        "duplicate_lines_removed": 0,
        "conflict_boxes_resolved": 0,
        "boxes_before": 0,
        "boxes_after": 0,
        "file_changed": False,
    }

    if not label_path.exists():
        return stats

    with open(label_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not raw_lines:
        return stats

    stats["boxes_before"] = len(raw_lines)

    bbox_to_class_counts: dict[tuple[float, float, float, float], dict[int, int]] = {}
    first_seen_order: list[tuple[float, float, float, float]] = []
    exact_seen: set[tuple[int, float, float, float, float]] = set()

    for line in raw_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = [round(float(v), 6) for v in parts[1:]]
        exact_key = (cls_id, cx, cy, w, h)
        bbox_key = (cx, cy, w, h)

        if exact_key in exact_seen:
            stats["duplicate_lines_removed"] += 1
            continue
        exact_seen.add(exact_key)

        if bbox_key not in bbox_to_class_counts:
            bbox_to_class_counts[bbox_key] = {}
            first_seen_order.append(bbox_key)
        bbox_to_class_counts[bbox_key][cls_id] = bbox_to_class_counts[bbox_key].get(cls_id, 0) + 1

    cleaned_lines: list[str] = []
    for bbox_key in first_seen_order:
        class_counts = bbox_to_class_counts[bbox_key]
        if len(class_counts) > 1:
            stats["conflict_boxes_resolved"] += 1

        chosen_class = min(
            class_counts.keys(),
            key=lambda cid: (-class_counts[cid], cid),
        )
        cx, cy, w, h = bbox_key
        cleaned_lines.append(f"{chosen_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    stats["boxes_after"] = len(cleaned_lines)
    stats["file_changed"] = cleaned_lines != raw_lines

    if stats["file_changed"]:
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines))

    return stats


def _sanitize_split_labels(target_labels_dir: Path) -> dict:
    aggregate = {
        "files_scanned": 0,
        "files_changed": 0,
        "duplicate_lines_removed": 0,
        "conflict_boxes_resolved": 0,
        "boxes_before": 0,
        "boxes_after": 0,
    }

    for label_file in sorted(target_labels_dir.glob("*.txt")):
        aggregate["files_scanned"] += 1
        file_stats = _sanitize_label_file_inplace(label_file)
        if file_stats["file_changed"]:
            aggregate["files_changed"] += 1
        aggregate["duplicate_lines_removed"] += file_stats["duplicate_lines_removed"]
        aggregate["conflict_boxes_resolved"] += file_stats["conflict_boxes_resolved"]
        aggregate["boxes_before"] += file_stats["boxes_before"]
        aggregate["boxes_after"] += file_stats["boxes_after"]

    return aggregate


def _write_data_yaml(dataset_path: Path, class_names: list[str]) -> Path:
    payload = {
        "path": str(dataset_path.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# Otomatik uretildi: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return yaml_path


def prepare_dataset(dataset_version: str) -> tuple[Path, dict]:
    """
    Dataset'i dogrular, 70/10/20 oraninda ayirir ve data.yaml uretir.
    Oranlar ve seed config/training_config.yaml altindan degistirilebilir.
    """
    dataset_path = DATASET_DIR / dataset_version
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset path not found: {dataset_path}")

    class_names: list[str] = config.get(
        "training", "class_names", default=["defect_scratch", "defect_crack", "defect_dent", "ok"]
    )

    train_ratio = float(config.get("training", "dataset_split", "train_ratio", default=0.70))
    val_ratio = float(config.get("training", "dataset_split", "val_ratio", default=0.10))
    test_ratio = float(config.get("training", "dataset_split", "test_ratio", default=0.20))
    seed = int(config.get("training", "dataset_split", "seed", default=42))
    clear_output_dirs = bool(config.get("training", "dataset_split", "clear_output_dirs", default=True))
    min_valid_samples = int(config.get("training", "dataset_split", "min_valid_samples", default=1))
    min_test_samples = int(config.get("training", "dataset_split", "min_test_samples", default=2))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise RuntimeError(
            f"Split ratio toplamı 1.0 olmalı. Mevcut toplam={ratio_sum:.6f}"
        )

    pairs, validation_report = _collect_pairs(dataset_path, class_names)
    if not pairs:
        raise RuntimeError("No valid image-label pairs found for training")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    train_n, val_n, test_n = _split_counts(
        len(pairs),
        train_ratio,
        val_ratio,
        test_ratio,
        min_valid_samples=min_valid_samples,
        min_test_samples=min_test_samples,
    )

    train_pairs = pairs[:train_n]
    valid_pairs = pairs[train_n:train_n + val_n]
    test_pairs = pairs[train_n + val_n:]

    targets = _prepare_output_dirs(dataset_path, clear_output_dirs)
    _copy_pairs(train_pairs, *targets["train"])
    _copy_pairs(valid_pairs, *targets["valid"])
    _copy_pairs(test_pairs, *targets["test"])

    sanitize_enabled = bool(
        config.get("training", "dataset_split", "sanitize_labels", default=True)
    )
    label_sanitization = {
        "enabled": sanitize_enabled,
        "train": {},
        "valid": {},
        "test": {},
    }
    if sanitize_enabled:
        label_sanitization["train"] = _sanitize_split_labels(targets["train"][1])
        label_sanitization["valid"] = _sanitize_split_labels(targets["valid"][1])
        label_sanitization["test"] = _sanitize_split_labels(targets["test"][1])

    yaml_path = _write_data_yaml(dataset_path, class_names)

    split_report = {
        "dataset_path": str(dataset_path),
        "yaml_path": str(yaml_path),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "total": len(pairs),
            "train": len(train_pairs),
            "valid": len(valid_pairs),
            "test": len(test_pairs),
        },
        "validation": validation_report,
        "label_sanitization": label_sanitization,
    }

    logger.info(
        "Dataset prepared: total=%d train=%d valid=%d test=%d yaml=%s",
        len(pairs),
        len(train_pairs),
        len(valid_pairs),
        len(test_pairs),
        yaml_path,
    )

    return yaml_path, split_report
