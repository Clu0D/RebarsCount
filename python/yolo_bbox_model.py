from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from stardist.matching import matching_dataset

from stardist_model import StarDistModel


class YoloBoundingBoxModel(StarDistModel):
    """Basic one-class YOLO bounding-box detector wrapped into the common model interface."""

    def __init__(
        self,
        n_channel_in,
        model_name="yolo_bbox",
        model_basedir="models",
        model_weights_file="weights_best.pt",
        train_batch_size=8,
        train_epochs=15,
        image_size=640,
        use_gpu=None,
        yolo_weights_dir="models/yolo_weights",
        yolo_weights_name="yolo11m.pt",
        conf_thresh=0.25,
        iou_thresh=0.45,
        class_name="SterjenArm",
    ):
        self.n_channel_in = int(n_channel_in)
        self.train_batch_size = int(train_batch_size)
        self.train_epochs = int(train_epochs)
        self.image_size = int(image_size)
        self.model_weights_file = str(model_weights_file)
        self.class_name = str(class_name)
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

        self._yolo_cls = None
        self._yolo = None

        if self._thresholds_path.exists():
            self._load_thresholds()

    @property
    def logdir(self) -> Path:
        return self._logdir

    def load_weights(self, weights_name: str | None = None):
        weights_path = self._logdir / (weights_name or self.model_weights_file)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO bbox weights not found: {weights_path}")
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
                    f"[YOLO-bbox] conf_thresh={float(conf_thresh):.3f}, "
                    f"iou_thresh={float(iou_thresh):.3f}, f1={score:.6f}"
                )
                if score > best_score:
                    best_score = score
                    best_thresholds = (float(conf_thresh), float(iou_thresh))

        self._conf_thresh, self._iou_thresh = best_thresholds
        self._save_thresholds()
        print(
            f"[YOLO-bbox] Using optimized values: "
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

        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            empty = np.zeros(image_u8.shape[:2], dtype=np.uint16)
            return empty, {"boxes_xyxy": [], "prob": []}

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        labels = self._boxes_to_label_mask(boxes_xyxy, image_u8.shape[:2])

        details = {
            "boxes_xyxy": boxes_xyxy.tolist(),
            "prob": scores.tolist(),
        }
        return labels, details

    def _get_yolo_cls(self):
        if self._yolo_cls is not None:
            return self._yolo_cls
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YoloBoundingBoxModel. Install project dependencies first."
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
            lines = self._labels_to_yolo_lines(np.asarray(label_mask, dtype=np.uint16))
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _labels_to_yolo_lines(self, label_mask: np.ndarray):
        height, width = label_mask.shape[:2]
        lines = []

        for obj_id in np.unique(label_mask):
            obj_id = int(obj_id)
            if obj_id == 0:
                continue
            ys, xs = np.where(label_mask == obj_id)
            if len(xs) == 0:
                continue

            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())

            bbox_w = float(x_max - x_min + 1)
            bbox_h = float(y_max - y_min + 1)
            x_center = float(x_min + x_max + 1) / 2.0
            y_center = float(y_min + y_max + 1) / 2.0

            lines.append(
                f"0 {x_center / width:.8f} {y_center / height:.8f} "
                f"{bbox_w / width:.8f} {bbox_h / height:.8f}"
            )

        return lines

    def _boxes_to_label_mask(self, boxes_xyxy: np.ndarray, shape: tuple[int, int]):
        height, width = shape
        labels = np.zeros((height, width), dtype=np.uint16)
        if boxes_xyxy.size == 0:
            return labels

        for obj_id, box in enumerate(boxes_xyxy, start=1):
            x1, y1, x2, y2 = box.tolist()
            x1 = int(max(0, min(width - 1, math.floor(x1))))
            y1 = int(max(0, min(height - 1, math.floor(y1))))
            x2 = int(max(0, min(width - 1, math.ceil(x2) - 1)))
            y2 = int(max(0, min(height - 1, math.ceil(y2) - 1)))
            if x2 < x1 or y2 < y1:
                continue
            labels[y1:y2 + 1, x1:x2 + 1] = int(obj_id)
        return labels

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
