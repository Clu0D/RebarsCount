package anton.axenov

/**
 * Selects one configured prediction pipeline for both zone and point tasks.
 */
enum class SegmentationPredictionPipeline {
    PYTHON,
    YOLO_SEG_FOR_ZONES_AND_POINTS,
    YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS,
}

/**
 * Describes which segmentation task is being executed.
 */
enum class SegmentationPredictionTarget {
    ZONES,
    POINTS,
}

/**
 * One model-like predictor that handles exactly one segmentation target.
 *
 * This abstraction is narrower than [SegmentationPredictionProvider]: it does
 * not decide whether the caller wants zones or points. That routing is handled
 * by [SplitSegmentationPredictionProvider].
 */
interface SegmentationTargetPredictor {
    /**
     * Human-readable predictor name used in diagnostics.
     */
    val predictorName: String

    /**
     * Predicts segmentation instances for one already encoded image.
     *
     * @param imageBytes encoded image bytes.
     * @param filename logical image filename.
     * @return decoded segmentation prediction.
     */
    suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
    ): SegmentationPrediction

    /**
     * Releases resources owned by this predictor.
     */
    fun close()
}

/**
 * Routes zone and point requests to two dedicated target predictors.
 *
 * This keeps the external [SegmentationPredictionProvider] contract stable
 * while allowing different model combinations for zones and points.
 *
 * @param zonePredictor predictor used for zone detection.
 * @param pointPredictor predictor used for point detection.
 */
class SplitSegmentationPredictionProvider(
    private val zonePredictor: SegmentationTargetPredictor,
    private val pointPredictor: SegmentationTargetPredictor,
) : SegmentationPredictionProvider {
    /**
     * Delegates to the configured zone or point predictor.
     *
     * @param imageBytes encoded image bytes.
     * @param filename logical image filename.
     * @param zonePrediction true when zones must be predicted, false for points.
     * @return segmentation prediction returned by the selected target predictor.
     */
    override suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
        zonePrediction: Boolean,
    ): SegmentationPrediction {
        return selectedPredictor(zonePrediction).predict(
            imageBytes = imageBytes,
            filename = filename,
        )
    }

    /**
     * Closes all distinct target predictors.
     */
    override fun close() {
        closeDistinctPredictors(zonePredictor, pointPredictor)
    }

    /**
     * Returns the predictor that matches the current request kind.
     *
     * @param zonePrediction true for zones and false for points.
     * @return predictor chosen for the request.
     */
    private fun selectedPredictor(zonePrediction: Boolean): SegmentationTargetPredictor {
        return if (zonePrediction) zonePredictor else pointPredictor
    }
}

/**
 * Placeholder target predictor used until the real ONNX model implementation is wired in.
 *
 * The stub returns an empty prediction while preserving the current DTO shape
 * used by the rest of the application.
 *
 * @param predictorName diagnostic predictor name.
 */
open class StubSegmentationTargetPredictor(
    override val predictorName: String,
) : SegmentationTargetPredictor {
    /**
     * Returns an empty placeholder prediction.
     *
     * @param imageBytes encoded image bytes.
     * @param filename logical image filename.
     * @return empty prediction placeholder.
     */
    override suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
    ): SegmentationPrediction {
        return SegmentationPrediction(
            filename = filename,
            width = 0,
            height = 0,
            count = 0,
            instances = emptyList(),
        )
    }

    /**
     * Releases resources held by the stub predictor.
     */
    override fun close() = Unit
}

/**
 * Placeholder ONNX target predictor for `yolo-seg`.
 *
 * @param target segmentation task that this model is expected to serve.
 */
class StubYoloSegTargetPredictor(
    target: SegmentationPredictionTarget,
) : StubSegmentationTargetPredictor(
    predictorName = "stub-yolo-seg-${target.name.lowercase()}",
)

/**
 * Placeholder ONNX target predictor for `yolo-stardist`.
 *
 * @param target segmentation task that this model is expected to serve.
 */
class StubYoloStarDistTargetPredictor(
    target: SegmentationPredictionTarget,
) : StubSegmentationTargetPredictor(
    predictorName = "stub-yolo-stardist-${target.name.lowercase()}",
)

/**
 * Closes predictors exactly once even when both targets share the same instance.
 *
 * @param predictors predictors that should be closed.
 */
private fun closeDistinctPredictors(vararg predictors: SegmentationTargetPredictor) {
    val closed = mutableListOf<SegmentationTargetPredictor>()
    predictors.forEach { predictor ->
        if (closed.none { existing -> existing === predictor }) {
            predictor.close()
            closed += predictor
        }
    }
}
