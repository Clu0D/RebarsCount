from __future__ import annotations

import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from inference_service import DummyInferenceService, SegmentationInferenceService

app = FastAPI(title="Segmentation API", version="2.0.0")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _build_service(
        prefix: str,
        default_kind: str,
        default_model_basedir: str,
        default_model_name: str,
        default_result_root: str,
):
    if _env(f"{prefix}_USE_DUMMY", "false").lower() == "true":
        return DummyInferenceService(int(_env(f"{prefix}_DUMMY_BOX_SIZE", "32")))
    return SegmentationInferenceService(
        model_kind=_env(f"{prefix}_MODEL_KIND", default_kind),
        model_basedir=_env(f"{prefix}_MODEL_BASEDIR", default_model_basedir),
        model_name=_env(f"{prefix}_MODEL_NAME", default_model_name),
        weights_file=_env(f"{prefix}_WEIGHTS_FILE", ""),
        result_root=_env(f"{prefix}_RESULT_ROOT", default_result_root),
        n_rays=int(_env(f"{prefix}_N_RAYS", "32")),
        image_size=int(_env(f"{prefix}_IMAGE_SIZE", "640")),
        yolo_weights_name=_env(f"{prefix}_YOLO_WEIGHTS_NAME", "yolo11m.pt"),
        class_name=_env(f"{prefix}_CLASS_NAME", "SterjenArm"),
    )


service_points = _build_service(
    prefix="POINTS",
    default_kind="yolo_stardist",
    default_model_basedir="models",
    default_model_name="points_yolo_stardist",
    default_result_root="results/points",
)
service_zones = _build_service(
    prefix="ZONES",
    default_kind="yolo_seg",
    default_model_basedir="models",
    default_model_name="zones_yolo_seg",
    default_result_root="results/zones",
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "message": "Python segmentation service is online"}


def _crop_center_square(image_bytes: bytes, crop_size: int = 300) -> bytes:
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        if width < crop_size or height < crop_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image is too small for a {crop_size}x{crop_size} center crop: got {width}x{height}",
            )

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        cropped = image.crop((left, top, left + crop_size, top + crop_size))

        output = BytesIO()
        save_format = image.format or "PNG"
        cropped.save(output, format=save_format)
        return output.getvalue()


def _prepare_image_bytes(image_bytes: bytes, crop_center: bool) -> bytes:
    if not crop_center:
        return image_bytes
    return _crop_center_square(image_bytes)


@app.post("/predict_points")
async def predict_points(
        file: UploadFile = File(...),
        crop_center: bool = Query(False, description="Crop the image to the center 300x300 square before prediction."),
) -> dict[str, object]:
    image_bytes = _prepare_image_bytes(await file.read(), crop_center)
    try:
        return service_points.predict_bytes(
            image_bytes,
            filename=file.filename or "image",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict_zones")
async def predict_zones(
        file: UploadFile = File(...),
        crop_center: bool = Query(False, description="Crop the image to the center 300x300 square before prediction."),
) -> dict[str, object]:
    image_bytes = _prepare_image_bytes(await file.read(), crop_center)
    try:
        return service_zones.predict_bytes(
            image_bytes,
            filename=file.filename or "image",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
