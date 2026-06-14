from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from stardist.geometry import dist_to_coord, polygons_to_label_coord, star_dist
from stardist.nms import non_maximum_suppression
from stardist.matching import matching_dataset
from stardist.utils import edt_prob

from stardist_model import StarDistModel


class YoloStarDistModel(StarDistModel):
    """Frozen YOLO11m features with lightweight StarDist-style prediction heads."""

    def __init__(
            self,
            n_channel_in,
            model_name="yolo_stardist",
            model_basedir="models",
            model_weights_file="weights_best.h5",
            n_rays=32,
            grid=(2, 2),
            train_patch_size=(128, 128),
            train_batch_size=4,
            train_steps_per_epoch=100,
            train_epochs=150,
            use_gpu=None,
            yolo_weights_dir="models/yolo_weights",
            yolo_weights_name="yolo11m.pt",
            dist_loss_weight=0.2,
            learning_rate=1e-3,
            train_background_reg=1e-4,
            force_grayscale_input=True,
            onnx_export_path=None,
    ):
        self.n_channel_in = int(n_channel_in)
        self.n_rays = int(n_rays)
        self.grid = tuple(int(v) for v in grid)
        self.train_patch_size = tuple(int(v) for v in train_patch_size)
        self.train_batch_size = int(train_batch_size)
        self.train_steps_per_epoch = int(train_steps_per_epoch)
        self.train_epochs = int(train_epochs)
        self.model_weights_file = str(model_weights_file)
        self.dist_loss_weight = float(dist_loss_weight)
        self.learning_rate = float(learning_rate)
        self.train_background_reg = float(train_background_reg)
        self.force_grayscale_input = bool(force_grayscale_input)
        self._logdir = Path(model_basedir) / model_name
        self._logdir.mkdir(parents=True, exist_ok=True)

        self._weights_path = self._logdir / self.model_weights_file
        self._thresholds_path = self._logdir / "thresholds.json"
        self._yolo_weights_dir = Path(yolo_weights_dir)
        self._yolo_weights_name = str(yolo_weights_name)
        self._prob_thresh = 0.5
        self._nms_thresh = 0.4
        self._best_val_loss = float("inf")

        self._torch = None
        self._nn = None
        self._F = None
        self._device = None
        self._yolo = None
        self._prob_head = None
        self._dist_head = None
        self._optimizer = None
        self._detect_hook = None
        self._captured_detect_inputs = None
        self._use_gpu = use_gpu
        self._model_stride = 32
        self._onnx_export_path = Path(onnx_export_path) if onnx_export_path is not None else None

        if self._thresholds_path.exists():
            self._load_thresholds()
        if self._onnx_export_path is not None:
            self._export_from_constructor()

    @property
    def logdir(self) -> Path:
        return self._logdir

    def load_weights(self, weights_name: str | None = None):
        self._ensure_model_initialized()
        weights_path = self._logdir / (weights_name or self.model_weights_file)
        checkpoint = self._torch.load(str(weights_path), map_location=self._device)
        self._prob_head.load_state_dict(checkpoint["prob_head"])
        self._dist_head.load_state_dict(checkpoint["dist_head"])

        thresholds = checkpoint.get("thresholds", {})
        self._prob_thresh = float(thresholds.get("prob", self._prob_thresh))
        self._nms_thresh = float(thresholds.get("nms", self._nms_thresh))

        if self._thresholds_path.exists():
            self._load_thresholds()

        print(
            f"[YOLO-StarDist] Loaded weights from {weights_path} "
            f"(prob_thresh={self._prob_thresh:.4f}, nms_thresh={self._nms_thresh:.4f})."
        )

    def train(self, X_trn, Y_trn, validation_data, augmenter=None):
        self._ensure_model_initialized()
        X_val, Y_val = validation_data

        history = []
        for epoch in range(1, self.train_epochs + 1):
            train_metrics = self._run_epoch(X_trn, Y_trn, augmenter=augmenter, training=True)
            val_metrics = self._run_epoch(X_val, Y_val, augmenter=None, training=False)

            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_prob_loss": train_metrics["prob_loss"],
                    "train_dist_loss": train_metrics["dist_loss"],
                    "val_loss": val_loss,
                    "val_prob_loss": val_metrics["prob_loss"],
                    "val_dist_loss": val_metrics["dist_loss"],
                }
            )

            self._save_checkpoint(self._logdir / "weights_last.h5", epoch=epoch, val_loss=val_loss)
            if val_loss <= self._best_val_loss:
                self._best_val_loss = val_loss
                self._save_checkpoint(self._weights_path, epoch=epoch, val_loss=val_loss)
            print(f"[YOLO-StarDist] epoch {epoch}/{self.train_epochs} - "
                f"dist_dist_iou_metric: {train_metrics['dist_dist_iou_metric']:.4f} - "
                f"dist_loss: {train_metrics['dist_loss']:.4f} - "
                f"dist_relevant_mae: {train_metrics['dist_relevant_mae']:.4f} - "
                f"dist_relevant_mse: {train_metrics['dist_relevant_mse']:.4f} - "
                f"loss: {train_loss:.4f} - "
                f"prob_kld: {train_metrics['prob_kld']:.4f} - "
                f"prob_loss: {train_metrics['prob_loss']:.4f} - "
                f"val_dist_dist_iou_metric: {val_metrics['dist_dist_iou_metric']:.4f} - "
                f"val_dist_loss: {val_metrics['dist_loss']:.4f} - "
                f"val_dist_relevant_mae: {val_metrics['dist_relevant_mae']:.4f} - "
                f"val_dist_relevant_mse: {val_metrics['dist_relevant_mse']:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"val_prob_kld: {val_metrics['prob_kld']:.4f} - "
                f"val_prob_loss: {val_metrics['prob_loss']:.4f} - "
                f"learning_rate: {float(self._optimizer.param_groups[0]['lr']):.4e}"
            )

        if self._onnx_export_path is not None:
            self.load_weights(self.model_weights_file)
            self.export_onnx(self._onnx_export_path)
        return history


    def optimize_thresholds(self, X_val, Y_val):
        prob_candidates = np.linspace(0.1, 0.9, 9)
        nms_candidates = np.linspace(0.1, 0.9, 9)
        best_score = -float("inf")
        best_thresholds = (self._prob_thresh, self._nms_thresh)

        for nms_thresh in nms_candidates:
            for prob_thresh in prob_candidates:
                preds = []
                for image in X_val:
                    labels, _ = self.predict_instances(
                        image,
                        prob_thresh=float(prob_thresh),
                        nms_thresh=float(nms_thresh),
                    )
                    preds.append(labels)
                metrics = matching_dataset(Y_val, preds, thresh=0.5, by_image=False)
                score = float(metrics.panoptic_quality)
                print(f"prob_thresh: {float(prob_thresh)}, nms_thresh: {float(nms_thresh)} score: {float(metrics.panoptic_quality)}")
                if score > best_score:
                    best_score = score
                    best_thresholds = (float(prob_thresh), float(nms_thresh))

        self._prob_thresh, self._nms_thresh = best_thresholds
        self._save_thresholds()
        print(
            f"[YOLO-StarDist] Using optimized values: "
            f"prob_thresh={self._prob_thresh:.6f}, nms_thresh={self._nms_thresh:.6f}"
        )
        print(f"[YOLO-StarDist] Saved thresholds to {self._thresholds_path}")
        return {"prob": self._prob_thresh, "nms": self._nms_thresh, "score": best_score}

    def predict_instances(self, image: np.ndarray, n_tiles=None, prob_thresh=None, nms_thresh=None):
        self._ensure_model_initialized()
        self._prob_head.eval()
        self._dist_head.eval()

        prob_thresh = self._prob_thresh if prob_thresh is None else float(prob_thresh)
        nms_thresh = self._nms_thresh if nms_thresh is None else float(nms_thresh)

        prob_map, dist_map = self._predict_maps(np.asarray(image), n_tiles=n_tiles)
        points, scores, dists = non_maximum_suppression(
            dist_map,
            prob_map,
            grid=self.grid,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )

        labels = np.zeros(image.shape[:2], dtype=np.uint16)
        details = {"points": points, "prob": scores, "dist": dists}
        if len(points) == 0:
            return labels, details

        coord = dist_to_coord(dists, points)
        labels = polygons_to_label_coord(coord, image.shape[:2])
        details["coord"] = coord
        return labels.astype(np.uint16), details

    def _ensure_model_initialized(self):
        if self._torch is not None:
            return

        self._torch = torch
        self._nn = nn
        self._F = F

        if self._use_gpu is None:
            use_cuda = torch.cuda.is_available()
        else:
            use_cuda = bool(self._use_gpu) and torch.cuda.is_available()
        if self._use_gpu is True and not use_cuda:
            print("[WARN] use_gpu=True requested for YOLO-StarDist, but CUDA is unavailable. Falling back to CPU.")
        self._device = torch.device("cuda" if use_cuda else "cpu")

        weights_path = self._ensure_yolo_weights(YOLO)
        self._yolo = YOLO(str(weights_path))
        self._yolo.model.to(self._device)
        self._yolo.model.eval()
        stride = getattr(self._yolo.model, "stride", None)
        if stride is not None:
            if hasattr(stride, "max"):
                self._model_stride = int(stride.max().item())
            elif isinstance(stride, (tuple, list)):
                self._model_stride = int(max(stride))
            else:
                self._model_stride = int(stride)
        for param in self._yolo.model.parameters():
            param.requires_grad = False

        detect_module = self._resolve_detect_module()
        self._detect_hook = detect_module.register_forward_pre_hook(self._capture_detect_inputs)

        feature_channels = 256
        self._prob_head = nn.Conv2d(feature_channels, 1, kernel_size=1).to(self._device)
        self._dist_head = nn.Conv2d(feature_channels, self.n_rays, kernel_size=1).to(self._device)
        self._optimizer = torch.optim.Adam(
            list(self._prob_head.parameters()) + list(self._dist_head.parameters()),
            lr=self.learning_rate,
        )

    def _export_from_constructor(self):
        if not self._weights_path.exists():
            raise FileNotFoundError(f"Can't find weights at {self._weights_path}.")
        self.load_weights(self.model_weights_file)
        self.export_onnx(self._onnx_export_path)

    def export_onnx(self, export_path: str | Path):
        self._ensure_model_initialized()
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        self._yolo.model.eval()
        self._prob_head.eval()
        self._dist_head.eval()

        export_h = self._round_up_to_stride(self.train_patch_size[0])
        export_w = self._round_up_to_stride(self.train_patch_size[1])
        dummy_input = self._torch.zeros((1, 3, export_h, export_w), dtype=self._torch.float32, device=self._device)

        outer_self = self

        class _ExportModule(self._nn.Module):
            def __init__(self):
                super().__init__()
                self.yolo_model = outer_self._yolo.model
                self.prob_head = outer_self._prob_head
                self.dist_head = outer_self._dist_head
                self.grid_y = int(outer_self.grid[0])
                self.grid_x = int(outer_self.grid[1])
                self._captured_detect_inputs = None
                self._detect_hook = self._resolve_detect_module().register_forward_pre_hook(self._capture_detect_inputs)

            def _resolve_detect_module(self):
                model = self.yolo_model
                if hasattr(model, "model") and len(model.model) > 0:
                    return model.model[-1]
                raise RuntimeError("Could not locate the YOLO detect module for ONNX export.")

            def _capture_detect_inputs(self, module, inputs):
                self._captured_detect_inputs = inputs

            def _extract_feature_map(self, x):
                self._captured_detect_inputs = None
                _ = self.yolo_model(x)

                if not self._captured_detect_inputs:
                    raise RuntimeError("YOLO detect hook did not capture any features during ONNX export.")

                features = self._captured_detect_inputs[0]
                if isinstance(features, tuple):
                    features = list(features)
                if not isinstance(features, list):
                    features = [features]

                feature_maps = [feat for feat in features if hasattr(feat, "ndim") and feat.ndim == 4]
                if not feature_maps:
                    raise RuntimeError("YOLO detect hook did not provide feature maps during ONNX export.")

                preferred = [feat for feat in feature_maps if int(feat.shape[1]) == 256]
                if preferred:
                    feature_maps = preferred
                feature_maps.sort(key=lambda feat: feat.shape[-1] * feat.shape[-2], reverse=True)
                feature = feature_maps[0]
                if int(feature.shape[1]) != 256:
                    raise RuntimeError(
                        f"Expected a 256-channel YOLO feature map for the StarDist heads, got {int(feature.shape[1])}."
                    )
                return feature

            def forward(self, x):
                feature = self._extract_feature_map(x)
                target_hw = (
                    int((x.shape[-2] + self.grid_y - 1) // self.grid_y),
                    int((x.shape[-1] + self.grid_x - 1) // self.grid_x),
                )
                prob_logits = outer_self._F.interpolate(
                    self.prob_head(feature),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                dist_map = outer_self._F.interpolate(
                    self.dist_head(feature),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                prob_map = outer_self._torch.sigmoid(prob_logits)
                dist_map = outer_self._torch.clamp(dist_map, min=0.0)
                return prob_map, dist_map

            def remove_hook(self):
                if self._detect_hook is not None:
                    self._detect_hook.remove()
                    self._detect_hook = None

        export_module = _ExportModule().to(self._device).eval()

        try:
            self._torch.onnx.export(
                export_module,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["prob_map", "dist_map"],
            )
        finally:
            export_module.remove_hook()

        metadata_path = export_path.with_suffix(".meta.json")
        metadata = {
            "model_type": "yolo_stardist",
            "onnx_path": export_path.name,
            "input_height": int(export_h),
            "input_width": int(export_w),
            "input_channels": 3,
            "n_rays": int(self.n_rays),
            "grid": [int(self.grid[0]), int(self.grid[1])],
            "prob_thresh": float(self._prob_thresh),
            "nms_thresh": float(self._nms_thresh),
            "output_layout": {
                "prob_map": "NCHW",
                "dist_map": "NCHW",
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[YOLO-StarDist] Exported ONNX model to {export_path}")
        print(f"[YOLO-StarDist] Saved export metadata to {metadata_path}")
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

    def _resolve_detect_module(self):
        model = self._yolo.model
        if hasattr(model, "model") and len(model.model) > 0:
            return model.model[-1]
        raise RuntimeError("Could not locate the YOLO detect module.")

    def _capture_detect_inputs(self, module, inputs):
        self._captured_detect_inputs = inputs

    def _extract_feature_map(self, image: np.ndarray):
        tensor = self._image_to_tensor(image)
        self._captured_detect_inputs = None
        with self._torch.no_grad():
            _ = self._yolo.model(tensor)

        if not self._captured_detect_inputs:
            raise RuntimeError("YOLO detect hook did not capture any features.")

        features = self._captured_detect_inputs[0]
        if isinstance(features, tuple):
            features = list(features)
        if not isinstance(features, list):
            features = [features]

        feature_maps = [feat for feat in features if hasattr(feat, "ndim") and feat.ndim == 4]
        if not feature_maps:
            raise RuntimeError("YOLO detect hook did not provide feature maps.")

        preferred = [feat for feat in feature_maps if int(feat.shape[1]) == 256]
        if preferred:
            feature_maps = preferred
        feature_maps.sort(key=lambda feat: feat.shape[-1] * feat.shape[-2], reverse=True)
        feature = feature_maps[0]
        if int(feature.shape[1]) != 256:
            raise RuntimeError(
                f"Expected a 256-channel YOLO feature map for the StarDist heads, got {int(feature.shape[1])}."
            )
        return feature

    def _predict_maps(self, image: np.ndarray, n_tiles=None):
        if n_tiles is not None and any(int(v) > 1 for v in n_tiles[:2]):
            return self._predict_maps_tiled(image, n_tiles=n_tiles)

        feature = self._extract_feature_map(image)
        target_hw = self._target_hw(image.shape[:2])
        with self._torch.no_grad():
            prob_logits = self._F.interpolate(
                self._prob_head(feature),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            dist_map = self._F.interpolate(
                self._dist_head(feature),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            prob_map = self._torch.sigmoid(prob_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            dist_map = dist_map[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        return prob_map, np.maximum(dist_map, 0.0)

    def _predict_maps_tiled(self, image: np.ndarray, n_tiles):
        tile_y = max(1, int(n_tiles[0]))
        tile_x = max(1, int(n_tiles[1]))
        h, w = image.shape[:2]

        out_h, out_w = self._target_hw((h, w))
        prob_sum = np.zeros((out_h, out_w), dtype=np.float32)
        dist_sum = np.zeros((out_h, out_w, self.n_rays), dtype=np.float32)
        weight_sum = np.zeros((out_h, out_w), dtype=np.float32)

        ys = np.linspace(0, h, tile_y + 1, dtype=int)
        xs = np.linspace(0, w, tile_x + 1, dtype=int)

        for yi in range(tile_y):
            for xi in range(tile_x):
                y0, y1 = ys[yi], ys[yi + 1]
                x0, x1 = xs[xi], xs[xi + 1]
                tile = image[y0:y1, x0:x1]
                tile_prob, tile_dist = self._predict_maps(tile, n_tiles=None)

                gy0 = y0 // self.grid[0]
                gx0 = x0 // self.grid[1]
                gy1 = gy0 + tile_prob.shape[0]
                gx1 = gx0 + tile_prob.shape[1]

                prob_sum[gy0:gy1, gx0:gx1] += tile_prob
                dist_sum[gy0:gy1, gx0:gx1] += tile_dist
                weight_sum[gy0:gy1, gx0:gx1] += 1.0

        weight_sum[weight_sum == 0] = 1.0
        prob_map = prob_sum / weight_sum
        dist_map = dist_sum / weight_sum[..., None]
        return prob_map, np.maximum(dist_map, 0.0)

    def _run_epoch(self, X, Y, augmenter, training):
        if training:
            self._prob_head.train()
            self._dist_head.train()
        else:
            self._prob_head.eval()
            self._dist_head.eval()

        losses = []
        prob_losses = []
        dist_losses = []
        prob_klds = []
        dist_relevant_maes = []
        dist_relevant_mses = []
        dist_dist_ious = []

        steps = self.train_steps_per_epoch if training else max(1, min(len(X), self.train_steps_per_epoch))
        for _ in range(steps):
            batch_x = []
            batch_prob = []
            batch_dist = []
            for _batch_idx in range(self.train_batch_size):
                sample_idx = random.randrange(len(X))
                image = np.asarray(X[sample_idx])
                labels = np.asarray(Y[sample_idx], dtype=np.uint16)

                if augmenter is not None:
                    image, labels = augmenter(image, labels)

                image, labels = self._random_patch(image, labels)
                prob_target, dist_target = self._make_targets(labels)

                batch_x.append(image)
                batch_prob.append(prob_target)
                batch_dist.append(dist_target)

            x_tensor = self._batch_images_to_tensor(batch_x)
            prob_target_t = self._torch.from_numpy(np.stack(batch_prob)[:, None]).to(self._device)
            dist_target_t = self._torch.from_numpy(np.stack(batch_dist)).permute(0, 3, 1, 2).to(self._device)

            with self._torch.set_grad_enabled(training):
                feature = self._extract_feature_map_from_tensor(x_tensor)
                target_hw = prob_target_t.shape[-2:]
                prob_logits = self._F.interpolate(
                    self._prob_head(feature),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                dist_pred = self._F.interpolate(
                    self._dist_head(feature),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )

                valid = prob_target_t >= 0
                prob_loss = self._F.binary_cross_entropy_with_logits(prob_logits[valid], prob_target_t[valid])
                prob_pred = self._torch.sigmoid(prob_logits)
                dist_loss = self._stardist_masked_mae_torch(
                    prob_target_t,
                    dist_target_t,
                    dist_pred,
                    reg_weight=self.train_background_reg,
                )
                dist_relevant_mae = self._stardist_masked_mae_torch(
                    prob_target_t,
                    dist_target_t,
                    dist_pred,
                    reg_weight=0.0,
                )
                dist_relevant_mse = self._stardist_masked_mse_torch(prob_target_t, dist_target_t, dist_pred)
                dist_dist_iou_metric = self._stardist_masked_iou_metric_torch(prob_target_t, dist_target_t, dist_pred)
                prob_kld = self._stardist_prob_kld_torch(prob_target_t, prob_pred)
                loss = prob_loss + self.dist_loss_weight * dist_loss

                if training:
                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self._optimizer.step()

            losses.append(float(loss.detach().cpu()))
            prob_losses.append(float(prob_loss.detach().cpu()))
            dist_losses.append(float(dist_loss.detach().cpu()))
            prob_klds.append(float(prob_kld.detach().cpu()))
            dist_relevant_maes.append(float(dist_relevant_mae.detach().cpu()))
            dist_relevant_mses.append(float(dist_relevant_mse.detach().cpu()))
            dist_dist_ious.append(float(dist_dist_iou_metric.detach().cpu()))

        return {
            "loss": float(np.mean(losses)),
            "prob_loss": float(np.mean(prob_losses)),
            "dist_loss": float(np.mean(dist_losses)),
            "prob_kld": float(np.mean(prob_klds)),
            "dist_relevant_mae": float(np.mean(dist_relevant_maes)),
            "dist_relevant_mse": float(np.mean(dist_relevant_mses)),
            "dist_dist_iou_metric": float(np.mean(dist_dist_ious)),
        }

    def _stardist_masked_mae_torch(self, dist_mask, dist_true, dist_pred, reg_weight=0.0):
        mask = dist_mask.float()
        abs_err = self._torch.abs(dist_true - dist_pred)
        actual_loss = (mask * abs_err).mean(dim=1)
        norm_mask = mask.mean(dim=(1, 2, 3)) + 1e-7
        loss = actual_loss.mean(dim=(1, 2)) / norm_mask
        if reg_weight > 0:
            reg_loss = ((1.0 - mask) * self._torch.abs(dist_pred)).mean(dim=1).mean(dim=(1, 2))
            loss = loss + float(reg_weight) * reg_loss
        return loss.mean()

    def _stardist_masked_mse_torch(self, dist_mask, dist_true, dist_pred):
        mask = dist_mask.float()
        sq_err = (dist_true - dist_pred) ** 2
        actual_loss = (mask * sq_err).mean(dim=1)
        norm_mask = mask.mean(dim=(1, 2, 3)) + 1e-7
        loss = actual_loss.mean(dim=(1, 2)) / norm_mask
        return loss.mean()

    def _stardist_masked_iou_metric_torch(self, dist_mask, dist_true, dist_pred):
        mask = dist_mask.float()
        pred_pos = self._torch.clamp(dist_pred, min=0.0)
        inter = self._torch.mean(self._torch.minimum(dist_true, pred_pos) ** 2, dim=1)
        union = self._torch.mean(self._torch.maximum(dist_true, pred_pos) ** 2, dim=1)
        iou = inter / (union + 1e-7)
        weighted = (mask[:, 0] * iou).mean(dim=(1, 2))
        norm_mask = mask[:, 0].mean(dim=(1, 2)) + 1e-7
        return (weighted / norm_mask).mean()

    def _stardist_prob_kld_torch(self, prob_true, prob_pred):
        valid = prob_true >= 0
        if not bool(valid.any()):
            return self._torch.tensor(0.0, device=prob_true.device, dtype=prob_true.dtype)
        eps = 1e-7
        y_true = self._torch.clamp(prob_true[valid], min=eps, max=1.0 - eps)
        y_pred = self._torch.clamp(prob_pred[valid], min=eps, max=1.0 - eps)
        bce_pred = self._F.binary_cross_entropy(y_pred, y_true, reduction="none")
        bce_true = self._F.binary_cross_entropy(y_true, y_true, reduction="none")
        return (bce_pred - bce_true).mean()

    def _extract_feature_map_from_tensor(self, tensor):
        self._captured_detect_inputs = None
        with self._torch.no_grad():
            _ = self._yolo.model(tensor)

        if not self._captured_detect_inputs:
            raise RuntimeError("YOLO detect hook did not capture any features during training.")

        features = self._captured_detect_inputs[0]
        if isinstance(features, tuple):
            features = list(features)
        if not isinstance(features, list):
            features = [features]

        feature_maps = [feat for feat in features if hasattr(feat, "ndim") and feat.ndim == 4]
        if not feature_maps:
            raise RuntimeError("YOLO detect hook did not provide feature maps during training.")
        preferred = [feat for feat in feature_maps if int(feat.shape[1]) == 256]
        if preferred:
            feature_maps = preferred
        feature_maps.sort(key=lambda feat: feat.shape[-1] * feat.shape[-2], reverse=True)
        feature = feature_maps[0]
        if int(feature.shape[1]) != 256:
            raise RuntimeError(
                f"Expected a 256-channel YOLO feature map for the StarDist heads, got {int(feature.shape[1])}."
            )
        return feature.detach()

    def _make_targets(self, labels: np.ndarray):
        prob = edt_prob(labels).astype(np.float32)
        prob = prob[:: self.grid[0], :: self.grid[1]]
        dist = star_dist(labels, self.n_rays, grid=self.grid).astype(np.float32)
        return prob, dist

    def _random_patch(self, image: np.ndarray, labels: np.ndarray):
        patch_h, patch_w = self.train_patch_size
        h, w = image.shape[:2]

        if h < patch_h or w < patch_w:
            pad_h = max(0, patch_h - h)
            pad_w = max(0, patch_w - w)
            if image.ndim == 2:
                image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
            else:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            labels = np.pad(labels, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = image.shape[:2]

        y0 = 0 if h == patch_h else random.randint(0, h - patch_h)
        x0 = 0 if w == patch_w else random.randint(0, w - patch_w)
        y1 = y0 + patch_h
        x1 = x0 + patch_w
        return image[y0:y1, x0:x1], labels[y0:y1, x0:x1]

    def _target_hw(self, hw):
        h, w = hw
        return int(math.ceil(h / self.grid[0])), int(math.ceil(w / self.grid[1]))

    def _image_to_tensor(self, image: np.ndarray):
        return self._batch_images_to_tensor([image])

    def _batch_images_to_tensor(self, images):
        batch = []
        prepared = [self._prepare_image_channels(image) for image in images]
        max_h = max(arr.shape[0] for arr in prepared)
        max_w = max(arr.shape[1] for arr in prepared)
        padded_h = self._round_up_to_stride(max_h)
        padded_w = self._round_up_to_stride(max_w)
        for arr in prepared:
            arr = self._pad_image_to_shape(arr, padded_h, padded_w)
            arr = np.transpose(arr, (2, 0, 1))
            batch.append(arr)

        tensor = self._torch.from_numpy(np.stack(batch)).to(self._device)
        return tensor.float()

    def _prepare_image_channels(self, image: np.ndarray):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if self.force_grayscale_input and arr.shape[-1] > 1:
            grayscale = np.mean(arr[..., :3], axis=-1, keepdims=True)
            arr = grayscale.astype(np.float32, copy=False)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] < 3:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, 3 - arr.shape[-1])), mode="edge")
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr

    def _pad_image_to_shape(self, arr: np.ndarray, padded_h: int, padded_w: int):
        pad_h = max(0, padded_h - arr.shape[0])
        pad_w = max(0, padded_w - arr.shape[1])
        if pad_h == 0 and pad_w == 0:
            return arr
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    def _round_up_to_stride(self, value: int):
        stride = max(1, int(self._model_stride))
        return int(math.ceil(value / stride) * stride)

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        checkpoint = {
            "prob_head": self._prob_head.state_dict(),
            "dist_head": self._dist_head.state_dict(),
            "thresholds": {"prob": self._prob_thresh, "nms": self._nms_thresh},
            "epoch": int(epoch),
            "val_loss": float(val_loss),
        }
        self._torch.save(checkpoint, str(path))

    def _load_thresholds(self):
        data = json.loads(self._thresholds_path.read_text())
        self._prob_thresh = float(data.get("prob", self._prob_thresh))
        self._nms_thresh = float(data.get("nms", self._nms_thresh))

    def _save_thresholds(self):
        payload = {"prob": self._prob_thresh, "nms": self._nms_thresh}
        self._thresholds_path.write_text(json.dumps(payload, indent=2) + "\n")
