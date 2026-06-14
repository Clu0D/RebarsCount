package anton.axenov

import java.io.File

/**
 * Builds one server-side prediction provider for the requested pipeline profile.
 *
 * @param pipeline selected predictor pipeline.
 * @param pythonBaseUrl Python fallback service base URL.
 * @param onnxModelDirectory repository directory that stores ONNX files.
 * @return provider used by the Ktor server.
 */
fun createServerPredictionProvider(
    pipeline: SegmentationPredictionPipeline = defaultServerPredictionPipeline(),
    pythonBaseUrl: String = PYTHON_SEGMENTATION_SERVER_URL,
    onnxModelDirectory: File = defaultServerOnnxModelDirectory(),
): SegmentationPredictionProvider {
    return when (pipeline) {
        SegmentationPredictionPipeline.PYTHON -> SplitSegmentationPredictionProvider(
            zonePredictor = SegmentationPredictor(
                baseUrl = pythonBaseUrl,
                target = SegmentationPredictionTarget.ZONES,
            ),
            pointPredictor = SegmentationPredictor(
                baseUrl = pythonBaseUrl,
                target = SegmentationPredictionTarget.POINTS,
            ),
        )

        SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_POINTS -> SplitSegmentationPredictionProvider(
            zonePredictor = createJvmOnnxTargetPredictor(
                modelSpec = OnnxModelCatalog.zonesYoloSeg,
                modelFile = onnxModelDirectory.resolve(OnnxModelRepositoryLayout.ZONES_YOLO_SEG_MODEL),
            ),
            pointPredictor = createJvmOnnxTargetPredictor(
                modelSpec = OnnxModelCatalog.pointsYoloSeg,
                modelFile = onnxModelDirectory.resolve(OnnxModelRepositoryLayout.POINTS_YOLO_SEG_MODEL),
            ),
        )

        SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS -> SplitSegmentationPredictionProvider(
            zonePredictor = createJvmOnnxTargetPredictor(
                modelSpec = OnnxModelCatalog.zonesYoloSeg,
                modelFile = onnxModelDirectory.resolve(OnnxModelRepositoryLayout.ZONES_YOLO_SEG_MODEL),
            ),
            pointPredictor = createJvmOnnxTargetPredictor(
                modelSpec = OnnxModelCatalog.pointsYoloStarDist,
                modelFile = onnxModelDirectory.resolve(OnnxModelRepositoryLayout.POINTS_YOLO_STARDIST_MODEL),
            ),
        )
    }
}

/**
 * Reads the desired server pipeline from JVM properties or environment variables.
 *
 * Supported values are enum names from [SegmentationPredictionPipeline].
 *
 * @return selected pipeline or
 * [SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS] by default.
 */
fun defaultServerPredictionPipeline(): SegmentationPredictionPipeline {
    val rawValue = System.getProperty(SERVER_PIPELINE_PROPERTY)
        ?: System.getenv(SERVER_PIPELINE_ENV)
        ?: return SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS
    return runCatching {
        SegmentationPredictionPipeline.valueOf(rawValue.trim().uppercase())
    }.getOrElse {
        SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS
    }
}

/**
 * Returns the filesystem directory that stores ONNX model files for the server.
 *
 * @return resolved ONNX model directory.
 */
fun defaultServerOnnxModelDirectory(): File {
    val configuredDirectory = System.getProperty(SERVER_ONNX_MODEL_DIR_PROPERTY)
        ?: System.getenv(SERVER_ONNX_MODEL_DIR_ENV)
        ?: OnnxModelRepositoryLayout.MODEL_DIRECTORY
    return File(configuredDirectory)
}

internal const val SERVER_PIPELINE_PROPERTY = "anton.axenov.segmentation.pipeline"
internal const val SERVER_PIPELINE_ENV = "SEGMENTATION_PIPELINE"
internal const val SERVER_ONNX_MODEL_DIR_PROPERTY = "anton.axenov.onnx.modelDir"
internal const val SERVER_ONNX_MODEL_DIR_ENV = "ONNX_MODEL_DIR"
internal const val PYTHON_SEGMENTATION_SERVER_URL = "http://127.0.0.1:8001"
