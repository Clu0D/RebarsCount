from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure
from stardist.matching import matching_dataset

from stardist_model import StarDistModel


class YoloSegmentationModel(StarDistModel):
    """Basic one-class YOLO instance segmentation detector wrapped into the common model interface."""

    def __init__(
        self,
        n_channel_in,
        model_name="yolo_seg",
        model_basedir="models",
        model_weights_file="weights_best.pt",
        train_batch_size=8,
        train_epochs=15,
        image_size=640,
        use_gpu=None,
        yolo_weights_dir="models/yolo_weights",
        yolo_weights_name="yolo11m-seg.pt",
        conf_thresh=0.25,
        iou_thresh=0.45,
        class_name="SterjenArm",
        max_polygon_points=128,
        onnx_export_path=None,
    ):
        self.n_channel_in = int(n_channel_in)
        self.train_batch_size = int(train_batch_size)
        self.train_epochs = int(train_epochs)
        self.image_size = int(image_size)
        self.model_weights_file = str(model_weights_file)
        self.class_name = str(class_name)
        self.max_polygon_points = int(max_polygon_points)
        self._logdir = Path(model_basedir) / model_name
        self._logdir.mkdir(parents=True, exist_ok=True)

        self._weights_path = self._logdir / self.model_weights_file
        self._last_weights_path = self._logdir / "weights_last.pt"
        self._thresholds_path = self._logdir / "thresholds.json"
        self._dataset_dir = self._logdir / "_yolo_dataset"
        self._yolo_weights_dir = Path(yolo_weights_dir)
        self._yolo_weights_name = str(yolo_weights_name)
        self._conf_thresh = float(conf_thresh)
        self._iou_thresh = float(iou_thresh)
        self._use_gpu = use_gpu
        self._onnx_export_path = Path(onnx_export_path) if onnx_export_path is not None else None

        self._yolo_cls = None
        self._yolo = None

        if self._thresholds_path.exists():
            self._load_thresholds()
        if self._onnx_export_path is not None:
            self._export_from_constructor()

    @property
    def logdir(self) -> Path:
        return self._logdir

    def load_weights(self, weights_name: str | None = None):
        weights_path = self._logdir / (weights_name or self.model_weights_file)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO seg weights not found: {weights_path}")
        self._ensure_yolo_loaded(weights_path=weights_path)
        if self._thresholds_path.exists():
            self._load_thresholds()

    def train(self, X_trn, Y_trn, validation_data, augmenter=None):
        X_val, Y_val = validation_data
        yolo_cls = self._get_yolo_cls()
        base_weights = self._ensure_yolo_weights(yolo_cls)
        data_yaml = self._prepare_yolo_dataset(X_trn, Y_trn, X_val, Y_val)

        model = yolo_cls(str(base_weights))
        train_kwargs = {
            "data": str(data_yaml),
            "epochs": self.train_epochs,
            "batch": self.train_batch_size,
            "imgsz": self.image_size,
            "project": str(self._logdir),
            "name": "training",
            "exist_ok": True,
            "verbose": True,
            "workers": 0,
            "plots": False,
            "save": True,
        }
        if self._use_gpu is not None:
            train_kwargs["device"] = 0 if self._use_gpu is True else "cpu"

        results = model.train(**train_kwargs)
        save_dir = getattr(results, "save_dir", None)
        if save_dir is None:
            save_dir = self._logdir / "training"
        save_dir = Path(save_dir)

        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"YOLO training did not produce best weights at {best_path}")

        shutil.copy2(best_path, self._weights_path)
        if last_path.exists():
            shutil.copy2(last_path, self._last_weights_path)

        self._ensure_yolo_loaded(weights_path=self._weights_path, force_reload=True)
        if self._onnx_export_path is not None:
            self.export_onnx(self._onnx_export_path)
        return results

    def optimize_thresholds(self, X_val, Y_val):
        conf_candidates = np.linspace(0.1, 0.9, 9)
        iou_candidates = np.linspace(0.1, 0.9, 9)
        best_score = -float("inf")
        best_thresholds = (self._conf_thresh, self._iou_thresh)

        for conf_thresh in conf_candidates:
            for iou_thresh in iou_candidates:
                preds = []
                for image in X_val:
                    labels, _ = self.predict_instances(
                        image,
                        conf_thresh=float(conf_thresh),
                        iou_thresh=float(iou_thresh),
                    )
                    preds.append(labels)

                metrics = matching_dataset(Y_val, preds, thresh=0.5, by_image=False)
                score = float(metrics.f1)
                print(
                    f"[YOLO-seg] conf_thresh={float(conf_thresh):.3f}, "
                    f"iou_thresh={float(iou_thresh):.3f}, f1={score:.6f}"
                )
                if score > best_score:
                    best_score = score
                    best_thresholds = (float(conf_thresh), float(iou_thresh))

        self._conf_thresh, self._iou_thresh = best_thresholds
        self._save_thresholds()
        print(
            f"[YOLO-seg] Using optimized values: "
            f"conf_thresh={self._conf_thresh:.6f}, iou_thresh={self._iou_thresh:.6f}"
        )
        return {"conf": self._conf_thresh, "iou": self._iou_thresh, "score": best_score}

    def predict_instances(self, image: np.ndarray, n_tiles=None, conf_thresh=None, iou_thresh=None):
        model = self._ensure_yolo_loaded()
        conf_thresh = self._conf_thresh if conf_thresh is None else float(conf_thresh)
        iou_thresh = self._iou_thresh if iou_thresh is None else float(iou_thresh)

        image_u8 = self._image_to_uint8(np.asarray(image))
        predict_kwargs = {
            "source": image_u8,
            "conf": conf_thresh,
            "iou": iou_thresh,
            "imgsz": self._round_up_image_size(max(image_u8.shape[:2])),
            "verbose": False,
        }
        if self._use_gpu is not None:
            predict_kwargs["device"] = 0 if self._use_gpu is True else "cpu"

        results = model.predict(**predict_kwargs)
        result = results[0]

        if getattr(result, "masks", None) is None or result.masks is None or len(result.masks.xy) == 0:
            empty = np.zeros(image_u8.shape[:2], dtype=np.uint16)
            return empty, {"polygons": [], "prob": []}

        polygons = [np.asarray(poly, dtype=np.float32) for poly in result.masks.xy]
        scores = []
        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32).tolist()
        labels = self._polygons_to_label_mask(polygons, image_u8.shape[:2])

        details = {
            "polygons": [poly.tolist() for poly in polygons],
            "prob": scores,
        }
        return labels, details

    def _get_yolo_cls(self):
        if self._yolo_cls is not None:
            return self._yolo_cls
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YoloSegmentationModel. Install project dependencies first."
            ) from exc
        self._yolo_cls = YOLO
        return self._yolo_cls

    def _ensure_yolo_loaded(self, weights_path: Path | None = None, force_reload: bool = False):
        if self._yolo is not None and not force_reload and weights_path is None:
            return self._yolo

        yolo_cls = self._get_yolo_cls()
        if weights_path is None:
            if self._weights_path.exists():
                weights_path = self._weights_path
            else:
                weights_path = self._ensure_yolo_weights(yolo_cls)

        self._yolo = yolo_cls(str(weights_path))
        return self._yolo

    def _export_from_constructor(self):
        if self._weights_path.exists():
            self.load_weights(self.model_weights_file)
        else:
            self._ensure_yolo_loaded()
        self.export_onnx(self._onnx_export_path)

    def export_onnx(self, export_path: str | Path):
        model = self._ensure_yolo_loaded()
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        exported = model.export(
            format="onnx",
            imgsz=self.image_size,
            batch=1,
            simplify=True,
            opset=17,
        )
        if isinstance(exported, (list, tuple)):
            exported = exported[0]
        exported_path = Path(exported)
        if exported_path.resolve() != export_path.resolve():
            shutil.copy2(exported_path, export_path)

        metadata_path = export_path.with_suffix(".meta.json")
        metadata = {
            "model_type": "yolo_seg",
            "onnx_path": export_path.name,
            "image_size": int(self.image_size),
            "conf_thresh": float(self._conf_thresh),
            "iou_thresh": float(self._iou_thresh),
            "class_name": self.class_name,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[YOLO-seg] Exported ONNX model to {export_path}")
        print(f"[YOLO-seg] Saved export metadata to {metadata_path}")
        return export_path

    def _ensure_yolo_weights(self, yolo_cls):
        self._yolo_weights_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            self._yolo_weights_dir / self._yolo_weights_name,
            Path(self._yolo_weights_name),
            Path.cwd() / self._yolo_weights_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                if candidate.parent != self._yolo_weights_dir:
                    target = self._yolo_weights_dir / self._yolo_weights_name
                    if not target.exists():
                        shutil.copy2(candidate, target)
                    return target
                return candidate

        model = yolo_cls(self._yolo_weights_name)
        target = self._yolo_weights_dir / self._yolo_weights_name

        downloaded = getattr(model, "ckpt_path", None)
        if downloaded:
            downloaded = Path(downloaded)
            if downloaded.exists() and downloaded != target:
                shutil.copy2(downloaded, target)
        elif Path(self._yolo_weights_name).exists():
            shutil.copy2(self._yolo_weights_name, target)

        if not target.exists():
            raise FileNotFoundError(
                f"Could not locate or download official YOLO weights '{self._yolo_weights_name}'."
            )
        return target

    def _prepare_yolo_dataset(self, X_trn, Y_trn, X_val, Y_val):
        if self._dataset_dir.exists():
            shutil.rmtree(self._dataset_dir)

        train_images_dir = self._dataset_dir / "images" / "train"
        val_images_dir = self._dataset_dir / "images" / "val"
        train_labels_dir = self._dataset_dir / "labels" / "train"
        val_labels_dir = self._dataset_dir / "labels" / "val"

        for path in (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._write_split(X_trn, Y_trn, train_images_dir, train_labels_dir, prefix="train")
        self._write_split(X_val, Y_val, val_images_dir, val_labels_dir, prefix="val")

        data_yaml = self._dataset_dir / "dataset.yaml"
        payload = (
            f"path: {self._dataset_dir.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"  0: {self.class_name}\n"
        )
        data_yaml.write_text(payload, encoding="utf-8")
        return data_yaml

    def _write_split(self, images, labels, images_dir: Path, labels_dir: Path, prefix: str):
        for idx, (image, label_mask) in enumerate(zip(images, labels)):
            image_u8 = self._image_to_uint8(np.asarray(image))
            image_path = images_dir / f"{prefix}_{idx:05d}.png"
            label_path = labels_dir / f"{prefix}_{idx:05d}.txt"

            Image.fromarray(image_u8).save(image_path)
            lines = self._labels_to_yolo_seg_lines(np.asarray(label_mask, dtype=np.uint16))
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _labels_to_yolo_seg_lines(self, label_mask: np.ndarray):
        height, width = label_mask.shape[:2]
        lines = []

        for obj_id in np.unique(label_mask):
            obj_id = int(obj_id)
            if obj_id == 0:
                continue
            polygon = self._largest_contour_polygon(label_mask == obj_id)
            if polygon is None or len(polygon) < 3:
                continue

            flat_points = []
            for x, y in polygon:
                flat_points.append(f"{float(x) / width:.8f}")
                flat_points.append(f"{float(y) / height:.8f}")
            lines.append("0 " + " ".join(flat_points))

        return lines

    def _largest_contour_polygon(self, binary_mask: np.ndarray):
        contours = measure.find_contours(binary_mask.astype(np.uint8), level=0.5)
        if not contours:
            return None

        best = None
        best_area = 0.0
        for contour in contours:
            if contour is None or len(contour) < 3:
                continue
            polygon = [(float(col), float(row)) for row, col in contour]
            if len(polygon) > 1 and polygon[0] == polygon[-1]:
                polygon = polygon[:-1]
            if len(polygon) < 3:
                continue
            area = self._polygon_area(polygon)
            if area > best_area:
                best_area = area
                best = polygon

        if best is None:
            return None
        return self._decimate_points(best, self.max_polygon_points)

    def _polygon_area(self, points):
        if len(points) < 3:
            return 0.0
        area = 0.0
        for idx, (x1, y1) in enumerate(points):
            x2, y2 = points[(idx + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _decimate_points(self, points, max_points: int):
        if len(points) <= max_points:
            return points
        step = int(math.ceil(len(points) / max_points))
        return points[::step]

    def _polygons_to_label_mask(self, polygons, shape: tuple[int, int]):
        height, width = shape
        mask_img = Image.new("I", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)

        for obj_id, polygon in enumerate(polygons, start=1):
            pts = []
            for x, y in np.asarray(polygon):
                pts.append((float(min(max(x, 0.0), width - 1)), float(min(max(y, 0.0), height - 1))))
            if len(pts) >= 3:
                draw.polygon(pts, fill=int(obj_id))

        return np.array(mask_img, dtype=np.uint16)

    def _image_to_uint8(self, image: np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _round_up_image_size(self, value: int):
        stride = 32
        return int(math.ceil(max(32, value) / stride) * stride)

    def _load_thresholds(self):
        payload = json.loads(self._thresholds_path.read_text(encoding="utf-8"))
        self._conf_thresh = float(payload.get("conf", self._conf_thresh))
        self._iou_thresh = float(payload.get("iou", self._iou_thresh))

    def _save_thresholds(self):
        payload = {"conf": self._conf_thresh, "iou": self._iou_thresh}
        self._thresholds_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
