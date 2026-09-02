# AR multi-frame rebar counter

This repository contains a research prototype for counting steel reinforcing bars in bundles. An Android operator walks around the stock, captures each bundle from several angles, and reviews the reconstructed result directly in an augmented-reality scene.

The main idea is to avoid treating every photo as an independent count. A single view often misses bar ends because of occlusion, uneven stacking, poor lighting, or limited physical access. The application therefore combines camera frames using ARCore tracking, segments bundles and individual bar ends, reconstructs observations in world coordinates, removes repeated detections, and reports a count for each bundle.

The project accompanies [master-thesis.pdf](./master-thesis.pdf), which describes the motivation, algorithms, and experimental evaluation in detail. The current repository is a prototype: Android contains the complete AR workflow; the desktop and iOS targets currently display fallback screens only.

## What the application does

During a counting session the Android application:

1. tracks the camera pose with ARCore and captures camera intrinsics and depth when available;
2. rejects frames with poor exposure, blur, unstable motion, or insufficient coverage;
3. detects bundle-shaped zones of interest;
4. keeps useful views of each zone from sufficiently different angles;
5. segments visible bar ends with YOLO segmentation or the hybrid YOLO–StarDist pipeline;
6. triangulates detections from different views, resolves duplicate hypotheses, and assigns reconstructed points to bundles;
7. overlays zones, suggested capture directions, reconstructed points, confidence, and per-bundle counts in AR;
8. lets the operator add or delete points, change a point's bundle, and delete an incorrectly detected zone.

Three processing modes are offered when the app starts:

| Mode | Inference location | Intended use |
| --- | --- | --- |
| Fully server-side | Ktor server | More powerful hardware and centralized processing; requires a reachable server. |
| Fully local | Android device through ONNX Runtime | Immediate operation without a network connection. |
| Deferred | Local during capture, with later upload to Ktor | Collect and review data offline, then request server processing when connectivity is available. In the current implementation, accepted frames are retained only for the active app session, not as a durable session archive. |

## Architecture

```text
Android camera + ARCore
        |
        v
frame quality filtering --> bundle segmentation --> multi-angle snapshot selection
                                                        |
                    +-----------------------------------+------------------+
                    |                                                      |
                    v                                                      v
          local ONNX Runtime                                      Ktor server :8000
                    |                                                      |
                    +-------------------+----------------------------------+
                                        v
                          shared fusion and session logic
                    triangulation -> deduplication -> bundle assignment
                                        |
                                        v
                              AR overlay and corrections

Optional Python service :8001
training, model export, and a fallback inference backend for the Ktor server
```

The Kotlin modules are organized as follows:

| Path | Responsibility |
| --- | --- |
| [`composeApp`](./composeApp) | Compose Multiplatform UI. Its Android source set contains the ARCore camera pipeline, scene rendering, local inference client, and operator controls. |
| [`shared`](./shared) | Cross-platform request models, ONNX post-processing, triangulation, hypothesis resolution, zone assignment, and session processing. |
| [`server`](./server) | Ktor API that keeps isolated in-memory state per client session and runs either JVM ONNX inference or the Python fallback. |
| [`python`](./python) | Dataset preparation, YOLO/StarDist training and evaluation, ONNX export, notebooks, and the optional FastAPI inference service. |
| [`iosApp`](./iosApp) | iOS host project; ARKit integration is not implemented. |

## Prerequisites

For the Kotlin application and server:

- Android Studio with Android SDK 36;
- JDK 17 or newer to run Gradle (the compiled Android/JVM bytecode target is Java 11);
- an Android device with a camera and ARCore support for the actual AR workflow;
- Filament `matc` version 1.68.2 to compile the custom rendering material;
- trained ONNX model files for whichever local or server pipeline you use.

For the optional Python environment:

- Docker with Compose;
- an NVIDIA GPU, compatible drivers, and NVIDIA Container Toolkit for the provided Compose configuration.

The Gradle wrapper downloads Gradle 8.14.3 and the remaining Kotlin dependencies automatically.

## Model files

Model weights are deliberately excluded from Git. Place exported models in `models/onnx/` using these exact names:

```text
models/onnx/
├── zones_yolo_seg.onnx
├── points_yolo_seg.onnx
└── points_yolo_stardist.onnx
```

The default Android and Ktor pipelines use `zones_yolo_seg.onnx` for bundles and `points_yolo_stardist.onnx` for bar ends. `points_yolo_seg.onnx` is needed only when selecting the all-YOLO-segmentation pipeline in code or through server configuration.

Models can be produced by the scripts under [`python`](./python). For example, from that directory:

```bash
python main.py --preset zones_yolo_seg --original-root /path/to/zones-coco --retrain-model
python main.py --preset points_yolo_stardist --original-root /path/to/points-coco --retrain-model
```

The preset names determine the expected architecture and export filename. Source datasets and trained weights are not included in this repository.

## Configure and run

### 1. Configure the Filament material compiler

Download Filament 1.68.2 and point Gradle to its `matc` executable in the untracked `local.properties` file:

```properties
filament.matc.path=/absolute/path/to/filament/bin/matc
```

You may instead set the `FILAMENT_MATC` environment variable or pass `-Pfilament.matc.path=...`. The build also checks the repository-local default `.tools/filament/1.68.2-host/filament/bin/matc`.

### 2. Configure the Android server address

Edit [`composeApp/src/androidMain/res/values/strings.xml`](./composeApp/src/androidMain/res/values/strings.xml) and replace the development IP address with the LAN address of the machine running the services:

```xml
<string name="segmentation_server_base_url">http://YOUR_HOST:8000</string>
<string name="python_segmentation_server_base_url">http://YOUR_HOST:8001</string>
```

Use `10.0.2.2` instead of `localhost` when connecting from the standard Android emulator. A physical device and server must be on mutually reachable networks. The Python URL is used only by a Python-backed local pipeline; the default local pipeline reads packaged ONNX assets.

### 3. Start the Ktor server

The default server pipeline loads the YOLO-segmentation bundle model and YOLO–StarDist point model from `models/onnx/`:

```bash
./gradlew :server:run
```

The server listens on all interfaces at port `8000`. Verify it with:

```bash
curl http://localhost:8000/health
```

Available prediction backends are selected with `SEGMENTATION_PIPELINE`:

```bash
SEGMENTATION_PIPELINE=YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS ./gradlew :server:run
SEGMENTATION_PIPELINE=YOLO_SEG_FOR_ZONES_AND_POINTS ./gradlew :server:run
SEGMENTATION_PIPELINE=PYTHON ./gradlew :server:run
```

Set `ONNX_MODEL_DIR=/another/directory` to load server models from a location other than `models/onnx`.

### 4. Install the Android app

The easiest route is to open the repository in Android Studio, select an ARCore-capable device, and run the `composeApp` Android configuration. From a shell:

```bash
./gradlew :composeApp:installDebug
```

Grant the camera permission when prompted. Choose the processing mode on the first screen, point the camera at the visible ends of the rebar bundles, enable zone detection, and move around each zone to collect distinct views. The on-screen controls can then start point recognition, review the full result, or correct individual zones and points.

## Optional Python service

The FastAPI service is a model-development and fallback inference path. It exposes `GET /health`, `POST /predict_zones`, and `POST /predict_points` on port `8001`; prediction requests use a multipart field named `file`.

The `api` and `jupyter` services share the `stardist-base` image, so build it first:

```bash
cd python
docker compose build jupyter
docker compose up api
```

The default API configuration expects trained Python model directories at `python/models/zones_yolo_seg` and `python/models/points_yolo_stardist`. For a smoke test without weights, start the API with dummy predictors:

```bash
docker compose run --service-ports --rm \
  -e ZONES_USE_DUMMY=true \
  -e POINTS_USE_DUMMY=true \
  api
```

Example request:

```bash
curl -X POST http://localhost:8001/predict_points -F "file=@test.png"
```

To use this service through Ktor, run the Python API first and start Ktor with `SEGMENTATION_PIPELINE=PYTHON`. The Ktor fallback currently expects the Python API at `http://127.0.0.1:8001`.

JupyterLab can be started with `docker compose up jupyter` and opened on port `8888`; the development token is `stardist`.

## Tests

Run the JVM test suites with:

```bash
./gradlew :shared:jvmTest :composeApp:jvmTest :server:test
```

The tests cover geometry and placement math, multi-frame component reconstruction, point/zone separation, session processing, and Ktor routes.

## Current limitations

- The complete AR interface is Android-only; desktop and iOS are placeholders.
- Model weights and datasets are not distributed with the repository.
- Server session state is held in memory and is lost when the process stops.
- Deferred capture is not yet a durable, restart-safe session history.
- Service URLs are Android string resources rather than runtime settings.
- The system is a research prototype, not a calibrated or certified inventory instrument.

The thesis identifies future work including a larger training dataset, stronger multi-view fusion, persistent session history, result export and audit logs, restored spatial anchors, and automatic model/parameter selection based on available compute.
