from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from csbdeep.utils import normalize
from stardist.matching import matching_dataset

from basic_stardist_model import BasicStarDistModel
from prepare_dataset import prepare_train_dataset, show_image_and_mask, show_image_and_mask_i
from results_to_coco import CocoExporter
from train import (
    DEFAULT_AUGMENT_PARAMS,
    load_image_raw,
    load_train_pairs,
    make_stardist_augmenter,
    normalize_train_images,
    print_runtime_info,
)
from yolo_bbox_model import YoloBoundingBoxModel
from yolo_seg_model import YoloSegmentationModel


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_OUTPUT_DIR = REPO_ROOT / "models" / "onnx"


@dataclass(frozen=True)
class ModelPreset:
    preset_name: str
    model_type: str
    model_name: str
    onnx_filename: str
    metrics_json_name: str
    yolo_model_name: str = "yolo11m.pt"
    yolo_epochs: int = 15
    yolo_stardist_n_rays: int = 32
    basic_stardist_n_rays: int = 12
    class_name: str = "SterjenArm"


MODEL_PRESETS = {
    "zones_yolo_seg": ModelPreset(
        preset_name="zones_yolo_seg",
        model_type="yolo_seg",
        model_name="zones_yolo_seg",
        onnx_filename="zones_yolo_seg.onnx",
        metrics_json_name="evaluation_metrics_zones_yolo_seg.json",
        yolo_model_name="yolo11m-seg.pt",
    ),
    "points_yolo_seg": ModelPreset(
        preset_name="points_yolo_seg",
        model_type="yolo_seg",
        model_name="points_yolo_seg",
        onnx_filename="points_yolo_seg.onnx",
        metrics_json_name="evaluation_metrics_points_yolo_seg.json",
        yolo_model_name="yolo11m-seg.pt",
    ),
    "points_yolo_stardist": ModelPreset(
        preset_name="points_yolo_stardist",
        model_type="yolo_stardist",
        model_name="points_yolo_stardist",
        onnx_filename="points_yolo_stardist.onnx",
        metrics_json_name="evaluation_metrics_points_yolo_stardist.json",
        yolo_model_name="yolo11m.pt",
        yolo_stardist_n_rays=32,
    ),
}


def canonical_model_type(model_type: str) -> str:
    return "yolo_stardist" if model_type == "yolo" else model_type


def split_indices(n_items: int, n_val: int, n_test: int, seed: int = 42):
    if n_items < 3:
        raise ValueError("At least 3 samples are required for train/val/test splitting.")

    n_val = max(1, min(int(n_val), n_items - 2))
    n_test = max(1, min(int(n_test), n_items - n_val - 1))
    if n_items - n_val - n_test < 1:
        n_test = max(1, n_items - n_val - 1)

    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)

    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val + n_test]
    train_idx = indices[n_val + n_test:]
    return train_idx, val_idx, test_idx


def select_items(items, indices):
    return [items[int(i)] for i in indices]


def compute_n_tiles_for_image(x: np.ndarray, max_tile_size: int = 1024):
    h, w = x.shape[:2]
    tile_y = max(1, int(math.ceil(h / max_tile_size)))
    tile_x = max(1, int(math.ceil(w / max_tile_size)))
    if x.ndim == 2:
        return (tile_y, tile_x)
    if x.ndim == 3:
        return (tile_y, tile_x, 1)
    raise ValueError(f"Unsupported image ndim for tiled prediction: {x.ndim}")


def count_instances(labels: np.ndarray) -> int:
    ids = np.unique(labels)
    return int(np.count_nonzero(ids))


def compute_foreground_metrics(y_true_list, y_pred_list):
    intersection = 0
    union = 0
    pred_foreground = 0
    true_foreground = 0

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        true_fg = np.asarray(y_true) > 0
        pred_fg = np.asarray(y_pred) > 0
        intersection += int(np.logical_and(true_fg, pred_fg).sum())
        union += int(np.logical_or(true_fg, pred_fg).sum())
        true_foreground += int(true_fg.sum())
        pred_foreground += int(pred_fg.sum())

    foreground_iou = float(intersection / union) if union > 0 else 1.0
    denom = true_foreground + pred_foreground
    foreground_dice = float((2 * intersection) / denom) if denom > 0 else 1.0
    return foreground_iou, foreground_dice


def compute_dataset_metrics(y_true_list, y_pred_list, latencies_ms, iou_thresh: float = 0.5):
    dataset_match = matching_dataset(y_true_list, y_pred_list, thresh=iou_thresh, by_image=False)
    foreground_iou, foreground_dice = compute_foreground_metrics(y_true_list, y_pred_list)

    true_counts = np.asarray([count_instances(y) for y in y_true_list], dtype=np.float64)
    pred_counts = np.asarray([count_instances(y) for y in y_pred_list], dtype=np.float64)
    count_diff = pred_counts - true_counts

    matched_mean_iou = float(getattr(dataset_match, "mean_matched_score", 0.0))
    matched_mean_dice = float((2.0 * matched_mean_iou) / (1.0 + matched_mean_iou)) if matched_mean_iou > 0 else 0.0

    metrics = {
        "foreground_iou": float(foreground_iou),
        "foreground_dice": float(foreground_dice),
        "matched_mean_iou": matched_mean_iou,
        "matched_mean_dice": matched_mean_dice,
        "precision": float(dataset_match.precision),
        "recall": float(dataset_match.recall),
        "f1": float(dataset_match.f1),
        "accuracy": float(dataset_match.accuracy),
        "tp": int(dataset_match.tp),
        "fp": int(dataset_match.fp),
        "fn": int(dataset_match.fn),
        "rmse_count": float(np.sqrt(np.mean(np.square(count_diff)))),
        "mae_count": float(np.mean(np.abs(count_diff))),
        "relative_error_count": float(np.mean(np.abs(count_diff) / np.maximum(true_counts, 1.0))),
        "signed_error_count": float(np.mean(count_diff)),
        "mean_true_count": float(np.mean(true_counts)),
        "mean_pred_count": float(np.mean(pred_counts)),
        "avg_inference_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "median_inference_ms": float(np.median(latencies_ms)) if latencies_ms else 0.0,
        "num_images": int(len(y_true_list)),
        "matching_iou_thresh": float(iou_thresh),
    }
    return metrics


def predict_dataset(model, images, split_name: str, max_tile_size: int):
    predictions = []
    latencies_ms = []

    for idx, image in enumerate(images, start=1):
        n_tiles = compute_n_tiles_for_image(np.asarray(image), max_tile_size=max_tile_size)
        started_at = time.perf_counter()
        pred_labels, _details = model.predict_instances(image, n_tiles=n_tiles)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        predictions.append(np.asarray(pred_labels, dtype=np.uint16))
        latencies_ms.append(float(elapsed_ms))
        print(f"[{split_name}] image {idx}/{len(images)} - inference {elapsed_ms:.2f} ms")

    return predictions, latencies_ms


def choose_best_matching_iou_thresh(
    y_true_list,
    y_pred_list,
    candidate_thresholds: list[float] | None = None,
):
    if candidate_thresholds is None:
        candidate_thresholds = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]

    best_thresh = None
    best_f1 = -float("inf")
    best_metrics = None

    for iou_thresh in candidate_thresholds:
        dataset_match = matching_dataset(y_true_list, y_pred_list, thresh=float(iou_thresh), by_image=False)
        score = float(dataset_match.f1)
        print(
            f"[matching-thresh-search] iou_thresh={float(iou_thresh):.2f}, "
            f"f1={score:.6f}, precision={float(dataset_match.precision):.6f}, "
            f"recall={float(dataset_match.recall):.6f}"
        )
        if score > best_f1:
            best_f1 = score
            best_thresh = float(iou_thresh)
            best_metrics = {
                "f1": float(dataset_match.f1),
                "precision": float(dataset_match.precision),
                "recall": float(dataset_match.recall),
                "tp": int(dataset_match.tp),
                "fp": int(dataset_match.fp),
                "fn": int(dataset_match.fn),
            }

    if best_thresh is None:
        raise RuntimeError("Could not select a matching IoU threshold.")

    return best_thresh, best_metrics


def evaluate_predictions(y_true_list, predictions, latencies_ms, split_name: str, iou_thresh: float):
    metrics = compute_dataset_metrics(y_true_list, predictions, latencies_ms, iou_thresh=iou_thresh)
    print_metrics(split_name, metrics)
    return metrics


def print_metrics(split_name: str, metrics: dict[str, object]):
    print(f"[{split_name}] foreground_iou={metrics['foreground_iou']:.6f}")
    print(f"[{split_name}] foreground_dice={metrics['foreground_dice']:.6f}")
    print(f"[{split_name}] matched_mean_iou={metrics['matched_mean_iou']:.6f}")
    print(f"[{split_name}] matched_mean_dice={metrics['matched_mean_dice']:.6f}")
    print(f"[{split_name}] precision={metrics['precision']:.6f}")
    print(f"[{split_name}] recall={metrics['recall']:.6f}")
    print(f"[{split_name}] f1={metrics['f1']:.6f}")
    print(f"[{split_name}] rmse_count={metrics['rmse_count']:.6f}")
    print(f"[{split_name}] mae_count={metrics['mae_count']:.6f}")
    print(f"[{split_name}] relative_error_count={metrics['relative_error_count']:.6f}")
    print(f"[{split_name}] signed_error_count={metrics['signed_error_count']:.6f}")
    print(f"[{split_name}] avg_inference_ms={metrics['avg_inference_ms']:.2f}")


def build_model(
    model_type: str,
    n_channel: int,
    model_basedir: str = "models",
    model_name: str | None = None,
    class_name: str = "SterjenArm",
    basic_stardist_n_rays: int = 12,
    basic_stardist_model_name: str = "stardist_rebar",
    yolo_stardist_n_rays: int = 32,
    yolo_model_name: str = "yolo11m.pt",
    yolo_epochs: int = 15,
    onnx_export_path: str = None,
):
    model_type = canonical_model_type(model_type)
    model_name = model_name or model_type

    if model_type == "basic":
        return BasicStarDistModel(
            n_channel_in=n_channel,
            model_name=model_name or basic_stardist_model_name,
            model_basedir=model_basedir,
            model_weights_file="weights_best.h5",
            n_rays=basic_stardist_n_rays,
            grid=(2, 2),
            train_patch_size=(128, 128),
            train_batch_size=8,
            train_steps_per_epoch=100,
            train_epochs=15,
        ), "weights_best.h5"

    if model_type == "yolo_stardist":
        from yolo_stardist_model import YoloStarDistModel

        return YoloStarDistModel(
            n_channel_in=n_channel,
            model_name=model_name,
            model_basedir=model_basedir,
            model_weights_file="weights_best.h5",
            n_rays=yolo_stardist_n_rays,
            grid=(2, 2),
            train_patch_size=(128, 128),
            train_batch_size=4,
            train_steps_per_epoch=100,
            train_epochs=yolo_epochs,
            use_gpu=False,
            force_grayscale_input=True,
            onnx_export_path=onnx_export_path
        ), "weights_best.h5"

    if model_type == "yolo_bbox":
        return YoloBoundingBoxModel(
            n_channel_in=n_channel,
            model_name=model_name,
            model_basedir=model_basedir,
            model_weights_file="weights_best.pt",
            train_batch_size=8,
            train_epochs=yolo_epochs,
            image_size=640,
            use_gpu=False,
            yolo_weights_name=yolo_model_name,
            class_name=class_name,
        ), "weights_best.pt"

    if model_type == "yolo_seg":
        seg_weights_name = (
            yolo_model_name if "-seg" in yolo_model_name else yolo_model_name.replace(".pt", "-seg.pt")
        )
        return YoloSegmentationModel(
            n_channel_in=n_channel,
            model_name=model_name,
            model_basedir=model_basedir,
            model_weights_file="weights_best.pt",
            train_batch_size=8,
            train_epochs=yolo_epochs,
            image_size=640,
            use_gpu=False,
            yolo_weights_name=seg_weights_name,
            class_name=class_name,
            onnx_export_path=onnx_export_path
        ), "weights_best.pt"

    raise ValueError(f"Unsupported model type: {model_type}")


def select_model_inputs(model_type: str, raw_images, normalized_images):
    return raw_images if canonical_model_type(model_type) in {"yolo_bbox", "yolo_seg"} else normalized_images


def export_todo_predictions(
        model,
        model_type: str,
        axis_norm,
        regenerate_result_masks: bool,
        max_tile_size: int,
        todo_root: str | Path,
        predicted_masks_root: str | Path,
        predicted_annotations_root: str | Path,
        class_name: str,
):
    if canonical_model_type(model_type) == "yolo_bbox":
        raise NotImplementedError(
            "Todo export is not implemented for plain YOLO bbox detectors."
        )

    TODO_ROOT = Path(todo_root)
    PREDICTED_MASKS_ROOT = Path(predicted_masks_root)
    PREDICTED_ANNOTATIONS_ROOT = Path(predicted_annotations_root)
    COCO_CATEGORY_NAME = class_name
    COCO_CATEGORY_ID = 1
    POLYGON_SIMPLIFY_TOLERANCE_REL = 0.07
    MAX_POLYGON_POINTS = 200
    MIN_INSTANCE_AREA_FOR_EXPORT = 8

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    todo_image_candidates = sorted(
        p for p in TODO_ROOT.glob("*/images/**/*") if p.is_file() and p.suffix.lower() in image_extensions
    )
    if not todo_image_candidates:
        raise FileNotFoundError("No image files found in dataset/todo/*/images/**")

    todo_image_path = todo_image_candidates[0]
    relative_to_todo = todo_image_path.relative_to(TODO_ROOT)
    todo_folder_name = relative_to_todo.parts[0]

    pred_dir = PREDICTED_MASKS_ROOT / todo_folder_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_path = pred_dir / f"{todo_image_path.stem}.tiff"

    if pred_mask_path.exists() and not regenerate_result_masks:
        print(f"Reusing existing preview mask: {pred_mask_path}")
    else:
        todo_raw = load_image_raw(todo_image_path)
        normalize_input = canonical_model_type(model_type) != "yolo_seg"
        todo_input = normalize(todo_raw, 1, 99.8, axis=axis_norm) if normalize_input else todo_raw
        n_tiles = compute_n_tiles_for_image(todo_input, max_tile_size=max_tile_size)
        labels, _details = model.predict_instances(todo_input, n_tiles=n_tiles)
        tifffile.imwrite(str(pred_mask_path), labels.astype("uint16"))
        print(f"Generated preview mask: {pred_mask_path} (n_tiles={n_tiles})")
        del todo_raw, todo_input, labels
        gc.collect()

    print(f"Preview image: {todo_image_path}")
    print(f"Preview mask: {pred_mask_path}")
    show_image_and_mask(todo_image_path, pred_mask_path)

    folders = sorted([p for p in TODO_ROOT.iterdir() if p.is_dir()])
    coco_exporter = CocoExporter()

    for folder_dir in folders:
        coco_exporter.predict_one_folder_to_coco(
            model,
            folder_dir,
            PREDICTED_MASKS_ROOT,
            PREDICTED_ANNOTATIONS_ROOT,
            COCO_CATEGORY_NAME,
            COCO_CATEGORY_ID,
            POLYGON_SIMPLIFY_TOLERANCE_REL,
            MAX_POLYGON_POINTS,
            max_tile_size,
            MIN_INSTANCE_AREA_FOR_EXPORT,
            axis_norm,
            normalize_input=canonical_model_type(model_type) != "yolo_seg",
            REGENERATE_RESULT_MASKS=regenerate_result_masks,
        )


def run_pipeline(
    model_type: str,
    original_root: str | Path = "dataset/original",
    train_root: str | Path = "dataset/train/default",
    models_root: str | Path = "models",
    allowed_labels: tuple[str, ...] | list[str] | None = ("SterjenArm",),
    validation_size: int = 30,
    test_size: int = 30,
    split_seed: int = 42,
    retrain_model: bool = True,
    regenerate_masks: bool = True,
    regenerate_result_masks: bool = True,
    show_preview: bool = False,
    export_todo_predictions_enabled: bool = False,
    matching_iou_thresh: float | None = None,
    max_tile_size: int = 1024,
    metrics_json_name: str = "evaluation_metrics.json",
    basic_stardist_n_rays: int = 12,
    basic_stardist_model_name: str = "stardist_rebar",
    yolo_stardist_n_rays: int = 32,
    yolo_model_name: str = "yolo11m.pt",
    yolo_epochs: int = 15,
    model_name: str | None = None,
    class_name: str = "SterjenArm",
    onnx_export_path: str = None,
    todo_root: str | Path = "dataset/todo",
    predicted_masks_root: str | Path = "dataset/predicted_masks",
    predicted_annotations_root: str | Path = "dataset/predicted_annotations",
):
    ORIGINAL_ROOT = Path(original_root)
    TRAIN_ROOT = Path(train_root)
    TRAIN_IMAGES_DIR = TRAIN_ROOT / "images"
    TRAIN_MASKS_DIR = TRAIN_ROOT / "generated_masks"
    MODELS_ROOT = Path(models_root)
    ALLOWED_LABELS = set(allowed_labels) if allowed_labels else None
    MIN_TRAIN_SIZE = (128, 128)

    model_type = canonical_model_type(model_type)
    model_name = model_name or model_type
    print(f"\n===== Running pipeline for {model_name} ({model_type}) =====")

    prepare_train_dataset(
        original_root=ORIGINAL_ROOT,
        train_images_dir=TRAIN_IMAGES_DIR,
        train_masks_dir=TRAIN_MASKS_DIR,
        allowed_labels=ALLOWED_LABELS,
        min_size=MIN_TRAIN_SIZE,
        regenerate_masks=regenerate_masks,
    )

    if show_preview:
        show_image_and_mask_i(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)

    print_runtime_info()

    X_raw_all, Y_all, names_all = load_train_pairs(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, min_train_size=MIN_TRAIN_SIZE)
    X_norm_all, n_channel, axis_norm = normalize_train_images(list(X_raw_all))

    train_idx, val_idx, test_idx = split_indices(
        len(X_raw_all),
        n_val=validation_size,
        n_test=test_size,
        seed=split_seed,
    )

    X_raw_trn = select_items(X_raw_all, train_idx)
    X_raw_val = select_items(X_raw_all, val_idx)
    X_raw_test = select_items(X_raw_all, test_idx)

    X_norm_trn = select_items(X_norm_all, train_idx)
    X_norm_val = select_items(X_norm_all, val_idx)
    X_norm_test = select_items(X_norm_all, test_idx)

    Y_trn = select_items(Y_all, train_idx)
    Y_val = select_items(Y_all, val_idx)
    Y_test = select_items(Y_all, test_idx)

    print(f"Total pairs: {len(X_raw_all)}")
    print(f"Train pairs: {len(X_raw_trn)}")
    print(f"Val pairs: {len(X_raw_val)}")
    print(f"Test pairs: {len(X_raw_test)}")
    print(f"n_channel_in: {n_channel}")

    model, weights_name = build_model(
        model_type=model_type,
        n_channel=n_channel,
        model_basedir=str(MODELS_ROOT),
        model_name=model_name,
        class_name=class_name,
        basic_stardist_n_rays=basic_stardist_n_rays,
        basic_stardist_model_name=basic_stardist_model_name,
        yolo_stardist_n_rays=yolo_stardist_n_rays,
        yolo_model_name=yolo_model_name,
        yolo_epochs=yolo_epochs,
        onnx_export_path=onnx_export_path
    )
    weights_path = model.logdir / weights_name

    train_images = select_model_inputs(model_type, X_raw_trn, X_norm_trn)
    val_images = select_model_inputs(model_type, X_raw_val, X_norm_val)
    test_images = select_model_inputs(model_type, X_raw_test, X_norm_test)

    augmenter = None
    if canonical_model_type(model_type) not in {"yolo_bbox", "yolo_seg"}:
        augmenter = make_stardist_augmenter(**DEFAULT_AUGMENT_PARAMS)
        print(f"Augmentation: enabled with params={DEFAULT_AUGMENT_PARAMS}")
    else:
        print("Augmentation: delegated to YOLO training defaults for bbox model.")

    if weights_path.exists() and not retrain_model:
        print(f"Reusing existing model weights: {weights_path}")
        model.load_weights(weights_name)
    else:
        print("Training started...")
        model.train(train_images, Y_trn, validation_data=(val_images, Y_val), augmenter=augmenter)
        print("Training finished.")

        print("Optimizing thresholds...")
        threshold_result = model.optimize_thresholds(val_images, Y_val)
        print(f"Threshold optimization finished: {threshold_result}")

    val_predictions, val_latencies_ms = predict_dataset(
        model,
        val_images,
        split_name="validation",
        max_tile_size=max_tile_size,
    )

    if matching_iou_thresh is None:
        selected_matching_iou_thresh, matching_thresh_search = choose_best_matching_iou_thresh(Y_val, val_predictions)
        print(
            f"[matching-thresh-search] selected_iou_thresh={selected_matching_iou_thresh:.2f}, "
            f"best_f1={matching_thresh_search['f1']:.6f}"
        )
    else:
        selected_matching_iou_thresh = float(matching_iou_thresh)
        matching_thresh_search = None
        print(f"[matching-thresh-search] using fixed_iou_thresh={selected_matching_iou_thresh:.2f}")

    val_metrics = evaluate_predictions(
        Y_val,
        val_predictions,
        val_latencies_ms,
        split_name="validation",
        iou_thresh=selected_matching_iou_thresh,
    )

    test_predictions, test_latencies_ms = predict_dataset(
        model,
        test_images,
        split_name="test",
        max_tile_size=max_tile_size,
    )
    test_metrics = evaluate_predictions(
        Y_test,
        test_predictions,
        test_latencies_ms,
        split_name="test",
        iou_thresh=selected_matching_iou_thresh,
    )

    metrics_payload = {
        "model_type": canonical_model_type(model_type),
        "weights_path": str(weights_path),
        "selected_matching_iou_thresh": float(selected_matching_iou_thresh),
        "matching_thresh_search": matching_thresh_search,
        "validation": val_metrics,
        "test": test_metrics,
        "split": {
            "train_size": len(X_raw_trn),
            "val_size": len(X_raw_val),
            "test_size": len(X_raw_test),
            "seed": split_seed,
        },
    }
    metrics_path = model.logdir / metrics_json_name
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")

    if export_todo_predictions_enabled:
        export_todo_predictions(
            model,
            model_type=model_type,
            axis_norm=axis_norm,
            regenerate_result_masks=regenerate_result_masks,
            max_tile_size=max_tile_size,
            todo_root=todo_root,
            predicted_masks_root=predicted_masks_root,
            predicted_annotations_root=predicted_annotations_root,
            class_name=class_name,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate and export segmentation models for the AR project.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Named target preset matching the Kotlin ONNX pipeline.",
    )
    parser.add_argument(
        "--model-type",
        choices=["basic", "yolo_stardist", "yolo_seg", "yolo_bbox", "yolo"],
        help="Model type to run when preset is not used.",
    )
    parser.add_argument(
        "--original-root",
        required=True,
        help="Path to the source COCO dataset root with subset folders.",
    )
    parser.add_argument(
        "--train-root",
        help="Work directory for prepared train images and masks.",
    )
    parser.add_argument(
        "--models-root",
        default="models",
        help="Directory where trained model folders and metrics are stored.",
    )
    parser.add_argument(
        "--onnx-output",
        help="Explicit ONNX output path. Defaults to models/onnx/<preset>.onnx for presets.",
    )
    parser.add_argument(
        "--allowed-label",
        action="append",
        dest="allowed_labels",
        help="COCO category name to keep. Repeat the flag for multiple labels.",
    )
    parser.add_argument("--validation-size", type=int, default=30)
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--yolo-model-name", default="yolo11m.pt")
    parser.add_argument("--yolo-epochs", type=int, default=15)
    parser.add_argument("--basic-stardist-n-rays", type=int, default=12)
    parser.add_argument("--yolo-stardist-n-rays", type=int, default=32)
    parser.add_argument("--metrics-json-name", default="evaluation_metrics.json")
    parser.add_argument("--model-name", help="Model folder name inside models root.")
    parser.add_argument("--class-name", default="SterjenArm")
    parser.add_argument("--matching-iou-thresh", type=float)
    parser.add_argument("--max-tile-size", type=int, default=1024)
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--retrain-model", action="store_true")
    parser.add_argument("--regenerate-masks", action="store_true")
    parser.add_argument("--regenerate-result-masks", action="store_true")
    parser.add_argument("--export-todo-predictions", action="store_true")
    parser.add_argument("--todo-root", default="dataset/todo")
    parser.add_argument("--predicted-masks-root", default="dataset/predicted_masks")
    parser.add_argument("--predicted-annotations-root", default="dataset/predicted_annotations")
    return parser


def resolve_run_configuration(args) -> dict[str, object]:
    preset = MODEL_PRESETS.get(args.preset) if args.preset else None
    if preset is None and not args.model_type:
        raise ValueError("Either --preset or --model-type must be provided.")

    model_type = preset.model_type if preset is not None else args.model_type
    default_model_name = (
        preset.model_name
        if preset is not None
        else ("stardist_rebar" if canonical_model_type(model_type) == "basic" else canonical_model_type(model_type))
    )
    model_name = args.model_name or default_model_name
    train_root = Path(args.train_root) if args.train_root else Path("dataset/train") / model_name
    onnx_output = args.onnx_output
    if onnx_output is None and preset is not None:
        onnx_output = str(DEFAULT_ONNX_OUTPUT_DIR / preset.onnx_filename)
    allowed_labels = (
        args.allowed_labels
        or ([preset.class_name] if preset is not None else ["SterjenArm"])
    )

    return {
        "model_type": model_type,
        "original_root": args.original_root,
        "train_root": str(train_root),
        "models_root": args.models_root,
        "allowed_labels": allowed_labels,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "split_seed": args.split_seed,
        "retrain_model": args.retrain_model,
        "regenerate_masks": args.regenerate_masks,
        "regenerate_result_masks": args.regenerate_result_masks,
        "show_preview": args.show_preview,
        "export_todo_predictions_enabled": args.export_todo_predictions,
        "matching_iou_thresh": args.matching_iou_thresh,
        "max_tile_size": args.max_tile_size,
        "metrics_json_name": args.metrics_json_name if preset is None else preset.metrics_json_name,
        "basic_stardist_n_rays": args.basic_stardist_n_rays if preset is None else preset.basic_stardist_n_rays,
        "yolo_stardist_n_rays": args.yolo_stardist_n_rays if preset is None else preset.yolo_stardist_n_rays,
        "yolo_model_name": args.yolo_model_name if preset is None else preset.yolo_model_name,
        "yolo_epochs": args.yolo_epochs if preset is None else preset.yolo_epochs,
        "model_name": model_name,
        "class_name": args.class_name if preset is None else preset.class_name,
        "onnx_export_path": onnx_output,
        "todo_root": args.todo_root,
        "predicted_masks_root": args.predicted_masks_root,
        "predicted_annotations_root": args.predicted_annotations_root,
    }


if __name__ == "__main__":
    parser = build_argument_parser()
    cli_args = parser.parse_args()
    run_pipeline(**resolve_run_configuration(cli_args))
