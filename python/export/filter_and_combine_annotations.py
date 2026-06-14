import json
from pathlib import Path


def normalize_name(name: str) -> set[str]:
    p = Path(name)
    return {
        p.name,
        p.with_suffix(".png").name,
        p.with_suffix(".jpg").name,
        p.with_suffix(".jpeg").name,
        p.stem,
    }


def build_allowed_names(raw_names: list[str]) -> tuple[set[str], set[str]]:
    exact_names: set[str] = set()
    stems: set[str] = set()
    for raw in raw_names:
        for variant in normalize_name(raw):
            exact_names.add(variant)
        stems.add(Path(raw).stem)
    return exact_names, stems


def image_is_allowed(file_name: str, exact_names: set[str], stems: set[str]) -> bool:
    if file_name in exact_names:
        return True
    return Path(file_name).stem in stems


def filter_coco(coco: dict, keep_file_names: list[str]) -> dict:
    exact_names, stems = build_allowed_names(keep_file_names)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    kept_images = [
        image for image in images
        if image_is_allowed(str(image.get("file_name", "")), exact_names, stems)
    ]
    kept_image_ids = {img.get("id") for img in kept_images}

    kept_annotations = [
        ann for ann in annotations
        if ann.get("image_id") in kept_image_ids
    ]

    filtered = dict(coco)
    filtered["images"] = kept_images
    filtered["annotations"] = kept_annotations
    return filtered


def _image_key_variants(file_name: str) -> list[str]:
    p = Path(file_name)
    return [p.name, p.stem]


def replace_segmentations_from_other_file(
        base_coco: dict,
        replacement_coco: dict,
) -> dict:
    """
    Replace annotations in base_coco with annotations from replacement_coco
    for images that match by file_name (fallback: stem).
    """
    base_images = base_coco.get("images", [])
    base_annotations = base_coco.get("annotations", [])
    replacement_images = replacement_coco.get("images", [])
    replacement_annotations = replacement_coco.get("annotations", [])

    # Match annotation key structure to the base COCO exported by CVAT.
    base_ann_has_attributes = any("attributes" in a for a in base_annotations)
    base_ann_has_ignore = any("ignore" in a for a in base_annotations)

    base_by_key: dict[str, dict] = {}
    for img in base_images:
        for key in _image_key_variants(str(img.get("file_name", ""))):
            base_by_key.setdefault(key, img)

    replacement_image_to_base_image_id: dict[int, int] = {}
    matched_base_ids: set[int] = set()

    for rep_img in replacement_images:
        rep_id = rep_img.get("id")
        if rep_id is None:
            continue
        match = None
        for key in _image_key_variants(str(rep_img.get("file_name", ""))):
            match = base_by_key.get(key)
            if match is not None:
                break
        if match is None:
            continue
        base_id = match.get("id")
        if base_id is None:
            continue
        replacement_image_to_base_image_id[rep_id] = base_id
        matched_base_ids.add(base_id)

    remaining_annotations = [
        ann for ann in base_annotations
        if ann.get("image_id") not in matched_base_ids
    ]

    next_ann_id = 1 + max(
        [int(a.get("id", 0)) for a in remaining_annotations if isinstance(a.get("id"), int)]
        or [0]
    )

    imported_annotations = []
    for ann in replacement_annotations:
        rep_image_id = ann.get("image_id")
        if rep_image_id not in replacement_image_to_base_image_id:
            continue
        new_ann = dict(ann)
        new_ann["image_id"] = replacement_image_to_base_image_id[rep_image_id]
        new_ann["id"] = next_ann_id
        next_ann_id += 1

        # Normalize schema fields to improve CVAT import compatibility.
        if base_ann_has_attributes:
            new_ann.setdefault("attributes", {})
            new_ann.pop("ignore", None)
        elif base_ann_has_ignore:
            new_ann.setdefault("ignore", 0)

        imported_annotations.append(new_ann)

    merged = dict(base_coco)
    merged["annotations"] = remaining_annotations + imported_annotations
    return merged


def main() -> None:
    directory = "job_668/annotations"
    coco_new_file = f"{directory}/coco_instance_segmentation.json"
    coco_new_filtered_file = f"{directory}/coco_instance_segmentation_filtered.json"
    coco_old_file = f"{directory}/instances_Train.json"
    coco_combined_file = f"{directory}/coco_combined.json"

    keep_filenames: list | None = None
    # keep_filenames = [
    #     "72a6da9f-9b6a-44b6-a669-eb2227bf12a9 (1)_region_0.json",
    #     "2025-07-09 11.33.58_region_2.png",
    #     "2025-07-09 11.34.01_region_28.png",
    #     "2025-07-09 11.34.01_region_29.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_1.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_2.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_3.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_10.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_11.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_16.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_17.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_24.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_26.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_28.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_29.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_30.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_31.png",
    #     "9800ba84-d063-4383-973b-3d3f4ea5ff1f_region_32.png",
    # ]

    with Path(coco_new_file).open("r", encoding="utf-8") as f:
        coco_new = json.load(f)

    if keep_filenames is None:
        coco_new_filtered = coco_new
    else:
        coco_new_filtered = filter_coco(coco_new, keep_filenames)

    with Path(coco_new_filtered_file).open("w", encoding="utf-8") as f:
        json.dump(coco_new_filtered, f, ensure_ascii=False, indent=2)

    print(f"Input images: {len(coco_new.get('images', []))}")
    print(f"Input annotations: {len(coco_new.get('annotations', []))}")
    print(f"Kept images: {len(coco_new_filtered.get('images', []))}")
    print(f"Kept annotations: {len(coco_new_filtered.get('annotations', []))}")
    print(f"Wrote: {coco_new_filtered_file}")

    with Path(coco_old_file).open("r", encoding="utf-8") as f:
        coco_old = json.load(f)

    coco_combined = replace_segmentations_from_other_file(coco_old, coco_new_filtered)

    with Path(coco_combined_file).open("w", encoding="utf-8") as f:
        json.dump(coco_combined, f, ensure_ascii=False, indent=2)

    print(f"Base annotations: {len(coco_old.get('annotations', []))}")
    print(f"Replacement annotations: {len(coco_new_filtered.get('annotations', []))}")
    print(f"Output annotations: {len(coco_combined.get('annotations', []))}")
    print(f"Wrote: {coco_combined_file}")
    return


if __name__ == "__main__":
    main()
