package anton.axenov

/**
 * Builds one server-side prediction provider for the requested pipeline profile.
 *
 * @param pipeline selected predictor pipeline.
 * @param pythonBaseUrl Python fallback service base URL.
 * @return provider used by the Ktor server.
 */
fun createServerPredictionProvider(
    pipeline: SegmentationPredictionPipeline = defaultServerPredictionPipeline(),
    pythonBaseUrl: String = PYTHON_SEGMENTATION_SERVER_URL,
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
            zonePredictor = StubYoloSegTargetPredictor(SegmentationPredictionTarget.ZONES),
            pointPredictor = StubYoloSegTargetPredictor(SegmentationPredictionTarget.POINTS),
        )

        SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS -> SplitSegmentationPredictionProvider(
            zonePredictor = StubYoloSegTargetPredictor(SegmentationPredictionTarget.ZONES),
            pointPredictor = StubYoloStarDistTargetPredictor(SegmentationPredictionTarget.POINTS),
        )
    }
}

/**
 * Reads the desired server pipeline from JVM properties or environment variables.
 *
 * Supported values are enum names from [SegmentationPredictionPipeline].
 *
 * @return selected pipeline or [SegmentationPredictionPipeline.PYTHON] by default.
 */
fun defaultServerPredictionPipeline(): SegmentationPredictionPipeline {
    val rawValue = System.getProperty(SERVER_PIPELINE_PROPERTY)
        ?: System.getenv(SERVER_PIPELINE_ENV)
        ?: return SegmentationPredictionPipeline.PYTHON
    return runCatching {
        SegmentationPredictionPipeline.valueOf(rawValue.trim().uppercase())
    }.getOrElse {
        SegmentationPredictionPipeline.PYTHON
    }
}

internal const val SERVER_PIPELINE_PROPERTY = "anton.axenov.segmentation.pipeline"
internal const val SERVER_PIPELINE_ENV = "SEGMENTATION_PIPELINE"
internal const val PYTHON_SEGMENTATION_SERVER_URL = "http://127.0.0.1:8001"
