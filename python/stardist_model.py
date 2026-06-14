from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

__all__ = ["StarDistModel"]


class StarDistModel(ABC):
    @property
    @abstractmethod
    def logdir(self) -> Path:
        pass

    @abstractmethod
    def load_weights(self, weights_name: str):
        pass

    @abstractmethod
    def train(self, X_trn, Y_trn, validation_data, augmenter=None):
        pass

    @abstractmethod
    def optimize_thresholds(self, X_val, Y_val):
        pass

    @abstractmethod
    def predict_instances(self, image: np.ndarray, n_tiles=None):
        pass

    def export_onnx(self, export_path: str | Path):
        raise NotImplementedError(f"{type(self).__name__} does not support ONNX export.")
