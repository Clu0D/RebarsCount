from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def parse_coco_images(coco_json_path: Path, allowed_labels=None):
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))

    categories = {int(c["id"]): c.get("name", "") for c in data.get("categories", [])}
    images_by_id = {int(img["id"]): img for img in data.get("images", [])}
    grouped = {img_id: [] for img_id in images_by_id.keys()}

    for ann in data.get("annotations", []):
        cat_name = categories.get(int(ann.get("category_id", -1)), "")
        if allowed_labels is not None and cat_name not in allowed_labels:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue

        seg = ann.get("segmentation", [])
        if not isinstance(seg, list):
            continue

        image_id = int(ann["image_id"])
        for poly_flat in seg:
            if not isinstance(poly_flat, list) or len(poly_flat) < 6:
                continue
            pts = []
            for i in range(0, len(poly_flat), 2):
                pts.append((float(poly_flat[i]), float(poly_flat[i + 1])))
            if len(pts) >= 3:
                grouped.setdefault(image_id, []).append(pts)

    items = []
    for image_id, img in images_by_id.items():
        file_name = img.get("file_name") or img.get("name")
        items.append(
            {
                "name": file_name,
                "width": int(img["width"]),
                "height": int(img["height"]),
                "polygons": grouped.get(image_id, []),
            }
        )

    return items


def find_original_subsets(original_root: Path):
    subsets = []
    if not original_root.exists():
        return subsets

    for subset_dir in sorted([p for p in original_root.iterdir() if p.is_dir()]):
        coco_json = subset_dir / "annotations" / "instances_default.json"
        images_dir = subset_dir / "images" / "default"
        if not coco_json.exists() or not images_dir.exists():
            print(f"[WARN] Skipping {subset_dir}: expected instances_default.json and images/default/ directory.")
            continue
        subsets.append((coco_json, images_dir))

    return subsets


def create_instance_mask(width: int, height: int, polygons):
    mask_img = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    obj_id = 1
    for polygon in polygons:
        draw.polygon(polygon, fill=obj_id)
        obj_id += 1

    return np.array(mask_img, dtype=np.uint32)


def resize_if_small(image_arr: np.ndarray, mask_arr: np.ndarray, min_size=(128, 128)):
    min_h, min_w = min_size
    h, w = image_arr.shape[:2]

    if h >= min_h and w >= min_w:
        return image_arr, mask_arr, False

    scale = max(min_h / h, min_w / w)
    new_h = int(np.ceil(h * scale))
    new_w = int(np.ceil(w * scale))

    image_pil = Image.fromarray(image_arr)
    mask_pil = Image.fromarray(mask_arr)

    image_resized = np.array(image_pil.resize((new_w, new_h), Image.Resampling.BILINEAR))
    mask_resized = np.array(mask_pil.resize((new_w, new_h), Image.Resampling.NEAREST), dtype=np.uint32)

    return image_resized, mask_resized, True


def build_dataset_item_filename(subset_name: str, image_name: str) -> str:
    source_path = Path(image_name)
    suffix = source_path.suffix or ".png"
    stem_parts = [subset_name, *source_path.with_suffix("").parts]
    safe_stem = "__".join(
        str(part).replace("\\", "_").replace("/", "_").replace(" ", "_")
        for part in stem_parts
        if str(part)
    )
    return f"{safe_stem}{suffix}"


def prepare_train_dataset(
    original_root: Path,
    train_images_dir: Path,
    train_masks_dir: Path,
    allowed_labels=None,
    min_size=(128, 128),
    regenerate_masks=True,
):
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_masks_dir.mkdir(parents=True, exist_ok=True)
    if regenerate_masks:
        for directory in (train_images_dir, train_masks_dir):
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_path.unlink()

    subsets = find_original_subsets(original_root)
    if not subsets:
        print(f"[WARN] No valid subsets found under: {original_root}")
        return

    copied = 0
    generated = 0
    collisions = 0
    resized = 0

    for coco_json_path, source_images_dir in subsets:
        subset_name = source_images_dir.parent.parent.name
        items = parse_coco_images(coco_json_path, allowed_labels=allowed_labels)

        for item in items:
            image_name = item["name"]
            src_image = source_images_dir / image_name
            dataset_item_name = build_dataset_item_filename(subset_name, image_name)
            dst_image = train_images_dir / dataset_item_name
            dst_mask = train_masks_dir / (Path(dataset_item_name).stem + ".tiff")

            if dst_image.exists() or dst_mask.exists():
                print(
                    f"[WARN] Target collision for '{dataset_item_name}' from subset '{subset_name}'. "
                    "Skipping this item."
                )
                collisions += 1
                continue

            if not src_image.exists():
                print(f"[WARN] Missing source image: {src_image}")
                continue

            image_arr = np.array(Image.open(src_image).convert("RGB"))
            mask_arr = create_instance_mask(item["width"], item["height"], item["polygons"])

            image_arr, mask_arr, was_resized = resize_if_small(image_arr, mask_arr, min_size=min_size)
            if was_resized:
                resized += 1

            Image.fromarray(image_arr).save(dst_image)
            copied += 1
            Image.fromarray(mask_arr).save(dst_mask, format="TIFF")
            generated += 1

    print(f"Copied images: {copied}")
    print(f"Generated masks: {generated}")
    print(f"Resized (small -> min_size): {resized}")
    print(f"Collisions skipped: {collisions}")


def show_image_and_mask(image_path: Path, mask_path: Path):
    image = Image.open(image_path).convert("RGB")
    mask = np.array(Image.open(mask_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Image: {image_path.name}")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="nipy_spectral")
    axes[1].set_title(f"Mask: {mask_path.name}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_image_and_mask_i(train_images_dir: Path, train_masks_dir: Path, i: int = 0):
    image_files = sorted([p for p in train_images_dir.iterdir() if p.is_file()])
    mask_files = sorted([p for p in train_masks_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])

    if not image_files:
        raise FileNotFoundError(f"No images found in {train_images_dir}")
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {train_masks_dir}")

    show_image_and_mask(image_files[i], mask_files[i])
