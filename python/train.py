from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_xla_fallback_flag = "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
if _xla_fallback_flag not in os.environ.get("XLA_FLAGS", ""):
    existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    os.environ["XLA_FLAGS"] = f"{existing_xla_flags} {_xla_fallback_flag}".strip()

import tensorflow as tf
import tifffile
from PIL import Image
from csbdeep.utils import normalize
from stardist import fill_label_holes, gputools_available

DEFAULT_AUGMENT_PARAMS = {
    "enable": True,
    "p_flip": 0.5,
    "p_rot90": 0.5,
    "p_perspective": 0.3,
    "perspective_scale_range": (0.01, 0.3),
    "p_affine": 0.5,
    "scale_x_range": (0.7, 1.3),
    "scale_y_range": (0.7, 1.3),
    "translate_x_percent_range": (-0.4, 0.4),
    "translate_y_percent_range": (-0.4, 0.4),
    "p_intensity": 0.5,
    "intensity_scale_range": (0.7, 1.3),
    "intensity_shift_range": (-0.1, 0.1),
    "p_hue": 0.5,
    "hue_shift_range": (-25, 25),
    "p_noise": 0.5,
    "noise_sigma_range": (0.0, 0.15),
    "seed": 42,
}


def print_runtime_info():
    try:
        import imgaug as ia  # noqa: F401
        import imgaug.augmenters as iaa  # noqa: F401
        from imgaug.augmentables.segmaps import SegmentationMapsOnImage  # noqa: F401

        imgaug_available = True
        imgaug_import_error = None
    except Exception as e:
        imgaug_available = False
        imgaug_import_error = repr(e)

    print("Kernel Python:", sys.executable)
    print("TensorFlow version:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))
    print("gputools available:", gputools_available())
    print("imgaug available:", imgaug_available)
    if not imgaug_available:
        print("imgaug import error:", imgaug_import_error)


def load_image_raw(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def resize_pair_if_small(x: np.ndarray, y: np.ndarray, min_size=(128, 128)):
    min_h, min_w = min_size
    h, w = x.shape[:2]

    if h >= min_h and w >= min_w:
        return x, y

    scale = max(min_h / h, min_w / w)
    new_h = int(np.ceil(h * scale))
    new_w = int(np.ceil(w * scale))

    x_img = Image.fromarray(x)
    y_img = Image.fromarray(y)

    x_resized = np.array(x_img.resize((new_w, new_h), Image.Resampling.BILINEAR))
    y_resized = np.array(y_img.resize((new_w, new_h), Image.Resampling.NEAREST), dtype=np.uint16)

    return x_resized, y_resized


def load_train_pairs(images_dir: Path, masks_dir: Path, min_train_size=(128, 128)):
    image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
    X, Y, names = [], [], []

    for image_path in image_files:
        mask_path = masks_dir / f"{image_path.stem}.tiff"
        if not mask_path.exists():
            continue

        x = load_image_raw(image_path)
        y = tifffile.imread(str(mask_path)).astype(np.uint16)
        y = fill_label_holes(y)
        x, y = resize_pair_if_small(x, y, min_size=min_train_size)

        X.append(x)
        Y.append(y)
        names.append(image_path.name)

    if not X:
        raise RuntimeError("No image/mask pairs found in dataset/train.")

    return X, Y, names


def split_train_val(X, Y, names, n_val=10):
    n = len(X)
    n_val = max(1, min(n_val, n - 1))
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)

    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    X_trn = [X[i] for i in trn_idx]
    Y_trn = [Y[i] for i in trn_idx]
    X_val = [X[i] for i in val_idx]
    Y_val = [Y[i] for i in val_idx]

    return X_trn, Y_trn, X_val, Y_val


def normalize_train_images(X_all):
    sample = X_all[0]
    if sample.ndim == 2:
        n_channel = 1
        axis_norm = (0, 1)
    else:
        n_channel = sample.shape[-1]
        axis_norm = (0, 1)

    X_all = [normalize(x, 1, 99.8, axis=axis_norm) for x in X_all]
    return X_all, n_channel, axis_norm


def make_stardist_augmenter(
    p_flip=0.5,
    p_rot90=0.5,
    p_perspective=0.2,
    perspective_scale_range=(0.01, 0.05),
    p_affine=0.5,
    scale_x_range=(0.9, 1.1),
    scale_y_range=(0.9, 1.1),
    translate_x_percent_range=(-0.1, 0.1),
    translate_y_percent_range=(-0.1, 0.1),
    p_intensity=0.5,
    intensity_scale_range=(0.9, 1.1),
    intensity_shift_range=(-0.08, 0.08),
    p_hue=0.35,
    hue_shift_range=(-20, 20),
    p_noise=0.35,
    noise_sigma_range=(0.0, 0.03),
    seed=42,
    enable=True,
):
    if not enable:
        return None

    try:
        import imgaug as ia
        import imgaug.augmenters as iaa
        from imgaug.augmentables.segmaps import SegmentationMapsOnImage
    except Exception as e:
        raise ImportError("imgaug is required for augmentation. Install it and re-run the import cell.") from e

    ia.seed(seed)

    geom_ops = [
        iaa.Flipud(p_flip),
        iaa.Fliplr(p_flip),
        iaa.Sometimes(p_rot90, iaa.Rot90((1, 3))),
        iaa.Sometimes(
            p_perspective,
            iaa.PerspectiveTransform(scale=perspective_scale_range, keep_size=True),
        ),
        iaa.Sometimes(
            p_affine,
            iaa.Affine(
                scale={"x": scale_x_range, "y": scale_y_range},
                translate_percent={"x": translate_x_percent_range, "y": translate_y_percent_range},
                order=1,
                mode="constant",
                cval=0,
            ),
        ),
    ]
    geom_aug = iaa.Sequential(geom_ops, random_order=True)

    add_shift_u8 = (
        int(np.round(intensity_shift_range[0] * 255.0)),
        int(np.round(intensity_shift_range[1] * 255.0)),
    )
    noise_sigma_u8 = (
        float(noise_sigma_range[0] * 255.0),
        float(noise_sigma_range[1] * 255.0),
    )

    def _photo_aug_for(x_aug):
        is_color = x_aug.ndim == 3 and x_aug.shape[-1] >= 3
        ops = []

        intensity_ops = []
        if intensity_scale_range is not None:
            intensity_ops.append(iaa.Multiply(intensity_scale_range, per_channel=False))
        if intensity_shift_range is not None and add_shift_u8 != (0, 0):
            intensity_ops.append(iaa.Add(add_shift_u8, per_channel=False))
        if intensity_ops:
            ops.append(iaa.Sometimes(p_intensity, iaa.Sequential(intensity_ops, random_order=True)))

        if is_color and p_hue > 0:
            ops.append(iaa.Sometimes(p_hue, iaa.AddToHueAndSaturation(hue_shift_range)))

        if noise_sigma_range is not None and noise_sigma_u8[1] > 0:
            ops.append(iaa.Sometimes(p_noise, iaa.AdditiveGaussianNoise(scale=noise_sigma_u8, per_channel=False)))

        if not ops:
            return None
        return iaa.Sequential(ops, random_order=True)

    def augmenter(x, y):
        x_aug = np.array(x, copy=True)
        y_aug = np.array(y, copy=True).astype(np.int32, copy=False)

        x_u8 = np.clip(x_aug * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)

        det_geom = geom_aug.to_deterministic()
        x_u8 = det_geom(image=x_u8)

        segmap = SegmentationMapsOnImage(y_aug, shape=x_u8.shape)
        y_aug = det_geom(segmentation_maps=segmap).get_arr()
        if y_aug.ndim == 3 and y_aug.shape[-1] == 1:
            y_aug = y_aug[..., 0]

        photo_aug = _photo_aug_for(x_u8)
        if photo_aug is not None:
            x_u8 = photo_aug(image=x_u8)

        x_aug = (np.asarray(x_u8, dtype=np.float32) / 255.0).clip(0.0, 1.0)
        y_aug = np.asarray(y_aug, dtype=np.uint16)
        return x_aug, y_aug

    return augmenter
