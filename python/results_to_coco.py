from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize
from shapely.geometry import Polygon
from skimage import measure
import tifffile
import gc

from train import load_image_raw


class CocoExporter:

    def __init__(self):
        pass

    def _poly_area(self, points):
        if len(points) < 3:
            return 0.0
        s = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    def _decimate_points(self, points, max_points=200):
        if len(points) <= max_points:
            return points
        step = int(math.ceil(len(points) / max_points))
        return points[::step]

    def simplify_polygon_points_shapely(self, points, tolerance_rel=0.1, max_points=32):
        if len(points) < 3:
            return None

        poly = Polygon(points)
        if poly.is_empty:
            return None

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            return None

        if hasattr(poly, "geoms"):
            geoms = [g for g in poly.geoms if not g.is_empty]
            if not geoms:
                return None
            poly = max(geoms, key=lambda g: g.area)

        minx, miny, maxx, maxy = poly.bounds
        diag = math.hypot(maxx - minx, maxy - miny)
        simplify_tolerance = tolerance_rel * diag
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        if poly.is_empty:
            return None

        if hasattr(poly, "geoms"):
            geoms = [g for g in poly.geoms if not g.is_empty]
            if not geoms:
                return None
            poly = max(geoms, key=lambda g: g.area)

        coords = list(poly.exterior.coords)
        if len(coords) < 4:
            return None

        pts = [(float(x), float(y)) for x, y in coords[:-1]]
        if len(pts) < 3:
            return None

        pts = self._decimate_points(pts, max_points=max_points)
        return pts if len(pts) >= 3 else None

    def instance_to_polygon(self, labels: np.ndarray, obj_id: int, tolerance_rel=0.07, max_points=200):
        binary = (labels == obj_id).astype(np.uint8)
        if binary.sum() == 0:
            return None

        contours = measure.find_contours(binary, level=0.5)
        if not contours:
            return None

        best = None
        best_area = 0.0
        for contour in contours:
            if contour is None or len(contour) < 3:
                continue
            pts = [(float(col), float(row)) for row, col in contour]
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                continue
            area = self._poly_area(pts)
            if area > best_area:
                best_area = area
                best = pts

        if best is None:
            return None

        return self.simplify_polygon_points_shapely(best, tolerance_rel=tolerance_rel, max_points=max_points)

    def compute_n_tiles_for_image(self, x: np.ndarray, max_tile_size=1024):
        h, w = x.shape[:2]
        tile_y = max(1, int(math.ceil(h / max_tile_size)))
        tile_x = max(1, int(math.ceil(w / max_tile_size)))

        if x.ndim == 2:
            return (tile_y, tile_x)
        if x.ndim == 3:
            return (tile_y, tile_x, 1)
        raise ValueError(f"Unsupported image ndim for tiled prediction: {x.ndim}")

    def predict_labels_tiled(
            self,
            model,
            raw: np.ndarray,
            axis_norm=(0, 1),
            max_tile_size=1024,
            normalize_input=True,
    ):
        model_input = normalize(raw, 1, 99.8, axis=axis_norm) if normalize_input else raw
        n_tiles = self.compute_n_tiles_for_image(model_input, max_tile_size=max_tile_size)
        labels, _details = model.predict_instances(model_input, n_tiles=n_tiles)
        return labels.astype(np.uint16), n_tiles

    def polygon_to_bbox(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

    def points_to_coco_segmentation(self, points, width, height):
        flat = []
        for x, y in points:
            x = min(max(x, 0.0), float(width - 1))
            y = min(max(y, 0.0), float(height - 1))
            flat.extend([float(round(x, 2)), float(round(y, 2))])
        return flat

    def labels_to_coco_annotations(self,
                                   labels: np.ndarray,
                                   image_id: int,
                                   ann_id_start: int,
                                   coco_category_id=1,
                                   min_instance_area_for_export=8,
                                   polygon_simplify_tolerance_rel=0.07,
                                   max_polygon_points=200,
                                   ):
        h, w = labels.shape[:2]
        anns = []
        ann_id = ann_id_start

        for obj_id in np.unique(labels):
            obj_id = int(obj_id)
            if obj_id == 0:
                continue

            pts = self.instance_to_polygon(
                labels,
                obj_id,
                tolerance_rel=polygon_simplify_tolerance_rel,
                max_points=max_polygon_points,
            )
            if pts is None:
                continue

            area_px = int((labels == obj_id).sum())
            if area_px < min_instance_area_for_export:
                continue

            segmentation = [self.points_to_coco_segmentation(pts, w, h)]
            if len(segmentation[0]) < 6:
                continue

            anns.append(
                {
                    "id": ann_id,
                    "image_id": int(image_id),
                    "category_id": int(coco_category_id),
                    "segmentation": segmentation,
                    "area": float(area_px),
                    "bbox": self.polygon_to_bbox(pts),
                    "iscrowd": 0,
                    "attributes": {},
                }
            )
            ann_id += 1

        return anns, ann_id

    def save_coco_1_0(self,
                      images,
                      annotations,
                      out_json_path: Path,
                      coco_category_name="SterjenArm",
                      coco_category_id=1,
                      ):
        payload = {
            "licenses": [],
            "info": {
                "description": "StarDist predicted annotations",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now(timezone.utc).isoformat(),
            },
            "categories": [
                {
                    "id": int(coco_category_id),
                    "name": str(coco_category_name),
                    "supercategory": "",
                }
            ],
            "images": images,
            "annotations": annotations,
        }
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def predict_one_folder_to_coco(self, model, folder_dir: Path,
                                   PREDICTED_MASKS_ROOT,
                                   PREDICTED_ANNOTATIONS_ROOT,
                                   COCO_CATEGORY_NAME,
                                   COCO_CATEGORY_ID,
                                   POLYGON_SIMPLIFY_TOLERANCE_REL,
                                   MAX_POLYGON_POINTS,
                                   PREDICT_MAX_TILE_SIZE,
                                   MIN_INSTANCE_AREA_FOR_EXPORT,
                                   axis_norm,
                                   normalize_input=True,
                                   REGENERATE_RESULT_MASKS=True
                                   ):
        images_dir = folder_dir / 'images' / 'default'
        if not images_dir.exists():
            print(f'[WARN] {folder_dir} has no images/ folder, skipping')
            return

        folder_name = folder_dir.name
        masks_out_dir = PREDICTED_MASKS_ROOT / folder_name
        ann_out_dir = PREDICTED_ANNOTATIONS_ROOT / folder_name
        masks_out_dir.mkdir(parents=True, exist_ok=True)
        ann_out_dir.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []
        next_ann_id = 1

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        image_paths = sorted([
            p for p in images_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in image_extensions
        ])

        if not image_paths:
            print(f'[WARN] No image files found under {images_dir}')
            return

        print(f'Folder {folder_name}: found {len(image_paths)} image file(s) for prediction')

        for idx, image_path in enumerate(image_paths, start=1):
            mask_path = masks_out_dir / f'{image_path.stem}.tiff'

            if mask_path.exists() and not REGENERATE_RESULT_MASKS:
                labels = tifffile.imread(str(mask_path)).astype(np.uint16)
                reused_mask = True
                n_tiles = None
            else:
                raw = load_image_raw(image_path)
                labels, n_tiles = self.predict_labels_tiled(
                    model,
                    raw,
                    axis_norm=axis_norm,
                    max_tile_size=PREDICT_MAX_TILE_SIZE,
                    normalize_input=normalize_input,
                )
                tifffile.imwrite(str(mask_path), labels)
                reused_mask = False
                del raw
                gc.collect()

            h, w = labels.shape[:2]
            coco_images.append({
                'id': idx,
                'width': int(w),
                'height': int(h),
                'file_name': image_path.name,
                'license': 0,
            })

            anns, next_ann_id = self.labels_to_coco_annotations(
                labels,
                image_id=idx,
                ann_id_start=next_ann_id,
                coco_category_id=COCO_CATEGORY_ID,
                min_instance_area_for_export=MIN_INSTANCE_AREA_FOR_EXPORT,
                polygon_simplify_tolerance_rel=POLYGON_SIMPLIFY_TOLERANCE_REL,
                max_polygon_points=MAX_POLYGON_POINTS,
            )
            coco_annotations.extend(anns)
            status = 'reused' if reused_mask else 'generated'
            tile_msg = '' if n_tiles is None else f', n_tiles={n_tiles}'
            print(
                f'[{idx}/{len(image_paths)}] {folder_name}/{image_path.name}: {status}{tile_msg}, instances={int(labels.max())}, coco_annotations={len(anns)} -> {mask_path.name}')
            del labels, anns
            gc.collect()

        out_json = ann_out_dir / 'instances_default.json'
        self.save_coco_1_0(
            coco_images,
            coco_annotations,
            out_json_path=out_json,
            coco_category_name=COCO_CATEGORY_NAME,
            coco_category_id=COCO_CATEGORY_ID,
        )

        predicted_mask_files = sorted(
            [p for p in masks_out_dir.iterdir() if p.is_file() and p.suffix.lower() == '.tiff'])

        expected_stems = {p.stem for p in image_paths}
        actual_stems = {p.stem for p in predicted_mask_files}
        missing_stems = sorted(expected_stems - actual_stems)

        print(
            f'Coverage check for folder {folder_name}: expected={len(expected_stems)}, masks={len(actual_stems)}, missed={len(missing_stems)}')
        if missing_stems:
            print('Missed files:')
            for stem in missing_stems:
                print(f'  - {stem}')

        print(f'Saved COCO JSON: {out_json}')
