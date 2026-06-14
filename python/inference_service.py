from __future__ import annotations

import json
import math
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from csbdeep.utils import normalize

_xla_fallback_flag = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
if _xla_fallback_flag not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '').strip()} {_xla_fallback_flag}".strip()

from basic_stardist_model import BasicStarDistModel
from results_to_coco import CocoExporter
from yolo_seg_model import YoloSegmentationModel
from yolo_stardist_model import YoloStarDistModel


def default_model_name_for_kind(model_kind: str) -> str:
    model_kind = model_kind.strip().lower()
    if model_kind == "basic":
        return "stardist_rebar"
    if model_kind == "yolo_seg":
        return "yolo_seg"
    if model_kind == "yolo_stardist":
        return "yolo_stardist"
    raise ValueError(f"Unsupported model kind: {model_kind}")


def default_weights_file_for_kind(model_kind: str) -> str:
    model_kind = model_kind.strip().lower()
    if model_kind == "yolo_seg":
        return "weights_best.pt"
    if model_kind in {"basic", "yolo_stardist"}:
        return "weights_best.h5"
    raise ValueError(f"Unsupported model kind: {model_kind}")


class SegmentationInferenceService:
    def __init__(
            self,
            model_kind: str,
            model_basedir: str,
            result_root: str | Path,
            model_name: str | None = None,
            weights_file: str | None = None,
            n_rays: int = 32,
            image_size: int = 640,
            use_gpu=None,
            yolo_weights_name: str | None = None,
            class_name: str = "SterjenArm",
    ) -> None:
        self.model_kind = model_kind.strip().lower()
        self.model_name = model_name or default_model_name_for_kind(self.model_kind)
        self.weights_file = weights_file or default_weights_file_for_kind(self.model_kind)
        self.result_root = Path(result_root)
        self._polygon_exporter = CocoExporter()
        self.model = self._build_model(
            model_basedir=model_basedir,
            n_rays=n_rays,
            image_size=image_size,
            use_gpu=use_gpu,
            yolo_weights_name=yolo_weights_name,
            class_name=class_name,
        )
        self.model.load_weights(self.weights_file)

    def _build_model(
            self,
            model_basedir: str,
            n_rays: int,
            image_size: int,
            use_gpu,
            yolo_weights_name: str | None,
            class_name: str,
    ):
        if self.model_kind == "basic":
            return BasicStarDistModel(
                n_channel_in=3,
                model_name=self.model_name,
                model_basedir=model_basedir,
                model_weights_file=self.weights_file,
                n_rays=n_rays,
                grid=(2, 2),
                train_patch_size=(128, 128),
                train_batch_size=8,
                train_steps_per_epoch=100,
                train_epochs=15,
            )

        if self.model_kind == "yolo_seg":
            return YoloSegmentationModel(
                n_channel_in=3,
                model_name=self.model_name,
                model_basedir=model_basedir,
                model_weights_file=self.weights_file,
                train_batch_size=8,
                train_epochs=15,
                image_size=image_size,
                use_gpu=use_gpu,
                yolo_weights_name=yolo_weights_name or "yolo11m-seg.pt",
                class_name=class_name,
            )

        if self.model_kind == "yolo_stardist":
            return YoloStarDistModel(
                n_channel_in=3,
                model_name=self.model_name,
                model_basedir=model_basedir,
                model_weights_file=self.weights_file,
                n_rays=n_rays,
                grid=(2, 2),
                train_patch_size=(128, 128),
                train_batch_size=4,
                train_steps_per_epoch=100,
                train_epochs=15,
                use_gpu=use_gpu,
                yolo_weights_name=yolo_weights_name or "yolo11m.pt",
                force_grayscale_input=True,
            )

        raise ValueError(f"Unsupported model kind: {self.model_kind}")

    def predict_bytes(self, image_bytes: bytes, filename: str) -> dict:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        raw = np.array(image)
        model_input = self._prepare_model_input(raw)
        n_tiles = self._compute_n_tiles(model_input)
        labels, details = self.model.predict_instances(model_input, n_tiles=n_tiles)

        instances = self._build_instances(labels, details)
        result_dir = self._prepare_result_dir(self.result_root, filename)
        result = {
            "filename": filename,
            "width": int(raw.shape[1]),
            "height": int(raw.shape[0]),
            "count": len(instances),
            "instances": instances,
        }
        self._save_prediction(result_dir, image, labels, filename, result)
        return result

    def _prepare_model_input(self, raw: np.ndarray) -> np.ndarray:
        if self.model_kind == "yolo_seg":
            return raw
        return normalize(raw, 1, 99.8, axis=(0, 1))

    def _compute_n_tiles(self, model_input: np.ndarray):
        h, w = model_input.shape[:2]
        tile_y = max(1, math.ceil(h / 1024))
        tile_x = max(1, math.ceil(w / 1024))
        if model_input.ndim == 2:
            return tile_y, tile_x
        return tile_y, tile_x, 1

    def _build_instances(self, labels: np.ndarray, details: dict) -> list[dict[str, object]]:
        if self.model_kind == "yolo_seg":
            return self._build_yolo_seg_instances(details)
        return self._build_label_instances(labels, details)

    def _build_yolo_seg_instances(self, details: dict) -> list[dict[str, object]]:
        polygons = details.get("polygons", [])
        scores = [float(score) for score in details.get("prob", [])]
        instances = []
        for index, polygon in enumerate(polygons, start=1):
            integer_polygon = self._sanitize_polygon_points(polygon)
            if len(integer_polygon) < 3:
                continue
            instances.append(
                {
                    "id": index,
                    "confidence": scores[index - 1] if index - 1 < len(scores) else 1.0,
                    "bbox": self._polygon_to_bbox(integer_polygon),
                    "polygon": integer_polygon,
                }
            )
        return instances

    def _build_label_instances(self, labels: np.ndarray, details: dict) -> list[dict[str, object]]:
        probabilities = self._extract_confidences(details)
        object_ids = [int(object_id) for object_id in np.unique(labels) if int(object_id) != 0]
        instances = []
        for index, object_id in enumerate(object_ids, start=1):
            polygon = self._polygon_exporter.instance_to_polygon(
                labels,
                object_id,
                tolerance_rel=0.07,
                max_points=200,
            )
            if polygon is None:
                continue
            integer_polygon = self._sanitize_polygon_points(polygon)
            if len(integer_polygon) < 3:
                continue
            instances.append(
                {
                    "id": object_id,
                    "confidence": probabilities[index - 1] if index - 1 < len(probabilities) else 1.0,
                    "bbox": self._polygon_to_bbox(integer_polygon),
                    "polygon": integer_polygon,
                }
            )
        return instances

    def _extract_confidences(self, details: dict) -> list[float]:
        probabilities = details.get("prob")
        if probabilities is None:
            return []
        return [float(probability) for probability in np.asarray(probabilities).reshape(-1)]

    def _sanitize_polygon_points(self, polygon) -> list[dict[str, int]]:
        points = []
        for point in polygon:
            x, y = point
            integer_point = {"x": int(round(float(x))), "y": int(round(float(y)))}
            if not points or points[-1] != integer_point:
                points.append(integer_point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        return points

    def _polygon_to_bbox(self, polygon: list[dict[str, int]]) -> dict[str, int]:
        xs = [point["x"] for point in polygon]
        ys = [point["y"] for point in polygon]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        return {
            "x": int(x_min),
            "y": int(y_min),
            "width": int(x_max - x_min + 1),
            "height": int(y_max - y_min + 1),
        }

    def _prepare_result_dir(self, result_root: str | Path, filename: str) -> Path:
        root = Path(result_root)
        stem = Path(filename).stem or "image"
        result_dir = root / stem
        if result_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result_dir = root / f"{stem}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        return result_dir

    def _save_prediction(
            self,
            result_dir: Path,
            image: Image.Image,
            labels: np.ndarray,
            filename: str,
            result: dict,
    ) -> dict[str, str]:
        original_ext = Path(filename).suffix or ".png"
        original_path = result_dir / f"original{original_ext}"
        mask_path = result_dir / "mask.tiff"
        preview_path = result_dir / "preview.png"
        result_json_path = result_dir / "result.json"

        image.save(original_path)
        Image.fromarray(labels.astype(np.uint16)).save(mask_path)
        self._save_preview(image, labels, preview_path, filename, mask_path.name)
        result_json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        return {
            "original": str(original_path),
            "mask": str(mask_path),
            "preview": str(preview_path),
            "result_json": str(result_json_path),
        }

    def _save_preview(
            self,
            image: Image.Image,
            labels: np.ndarray,
            preview_path: Path,
            image_name: str,
            mask_name: str,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image)
        axes[0].set_title(f"Image: {image_name}")
        axes[0].axis("off")

        axes[1].imshow(labels, cmap="nipy_spectral")
        axes[1].set_title(f"Mask: {mask_name}")
        axes[1].axis("off")

        plt.tight_layout()
        fig.savefig(preview_path, bbox_inches="tight")
        plt.close(fig)


class StarDistInferenceService(SegmentationInferenceService):
    def __init__(self, model_basedir: str, n_rays: int, result_root: str | Path) -> None:
        super().__init__(
            model_kind="basic",
            model_basedir=model_basedir,
            result_root=result_root,
            n_rays=n_rays,
        )


class DummyInferenceService:
    def __init__(self, box_size: int):
        self.box_size = box_size

    def predict_bytes(self, image_bytes: bytes, filename: str) -> dict:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        raw = np.array(image)
        width = int(raw.shape[1])
        height = int(raw.shape[0])

        x = int(width / 2 - self.box_size / 2)
        y = int(height / 2 - self.box_size / 2)
        polygon = [
            {"x": x, "y": y},
            {"x": x + int(self.box_size), "y": y},
            {"x": x + int(self.box_size), "y": y + int(self.box_size)},
            {"x": x, "y": y + int(self.box_size)},
        ]
        instances = [{
            "id": 1,
            "confidence": 1.0,
            "bbox": {
                "x": x,
                "y": y,
                "width": int(self.box_size),
                "height": int(self.box_size),
            },
            "polygon": polygon,
        }]

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "count": len(instances),
            "instances": instances,
        }
