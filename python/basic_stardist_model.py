from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from stardist import gputools_available
from stardist.models import Config2D, StarDist2D

from stardist_model import StarDistModel


class BasicStarDistModel(StarDistModel):
    def __init__(
        self,
        n_channel_in,
        model_name="stardist_rebar",
        model_basedir="models",
        model_weights_file="weights_best.h5",
        n_rays=32,
        grid=(2, 2),
        train_patch_size=(128, 128),
        train_batch_size=4,
        train_steps_per_epoch=100,
        train_epochs=150,
        use_gpu=None,
        disable_xla_jit=True,
        force_tf_cpu_when_stardist_cpu=True,
        onnx_export_path=None,
    ):
        if disable_xla_jit:
            # Avoid TF/XLA PTX toolchain requirements in container/runtime setups
            # where ptxas/nvlink are not installed.
            try:
                tf.config.optimizer.set_jit(False)
            except Exception:
                pass

        if use_gpu is None:
            use_gpu = len(tf.config.list_physical_devices("GPU")) > 0 and gputools_available()
        elif use_gpu:
            use_gpu = gputools_available()
            if not use_gpu:
                print("[WARN] use_gpu=True requested for StarDist, but gputools is unavailable. Falling back to CPU.")

        if force_tf_cpu_when_stardist_cpu and not use_gpu:
            # StarDist CPU mode alone does not prevent TensorFlow from selecting GPU.
            # Hide GPU devices so training doesn't fail in partial CUDA/XLA environments.
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception as e:
                print(f"[WARN] Could not force TensorFlow CPU mode: {e}")

        conf = Config2D(
            n_rays=n_rays,
            grid=grid,
            use_gpu=use_gpu,
            n_channel_in=n_channel_in,
            train_patch_size=train_patch_size,
            train_batch_size=train_batch_size,
            train_steps_per_epoch=train_steps_per_epoch,
            train_epochs=train_epochs,
        )

        self._model = StarDist2D(conf, name=model_name, basedir=model_basedir)
        self.model_weights_file = model_weights_file

        if onnx_export_path is not None:
            self.export_onnx(onnx_export_path)

    @property
    def logdir(self) -> Path:
        return Path(self._model.logdir)

    def load_weights(self, weights_name: str | None = None):
        self._model.load_weights(weights_name or self.model_weights_file)

    def train(self, X_trn, Y_trn, validation_data, augmenter=None):
        return self._model.train(X_trn, Y_trn, validation_data=validation_data, augmenter=augmenter)

    def optimize_thresholds(self, X_val, Y_val):
        return self._model.optimize_thresholds(X_val, Y_val)

    def predict_instances(self, image, n_tiles=None):
        return self._model.predict_instances(image, n_tiles=n_tiles)

    def export_onnx(self, export_path: str | Path):
        raise NotImplementedError("BasicStarDistModel ONNX export is not implemented.")