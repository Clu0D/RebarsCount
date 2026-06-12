package anton.axenov

/**
 * Thresholds used while triangulating observations and assembling multi-view components.
 *
 * @param correspondenceEpsilonMeters maximal distance between rays producing a support edge.
 * @param forbiddenEpsilonMeters distance after which a compared observation pair is forbidden.
 * @param minNormalizationConfidence raw edge confidence mapped to zero.
 * @param clusterRadiusMeters maximal distance from a support edge to a component center.
 * @param conflictConfidenceThreshold minimal normalized edge confidence treated as support.
 * @param additionConfidenceThreshold minimal average support required to add an observation.
 * @param componentConfidenceThreshold minimal aggregated confidence of a valid component.
 */
internal data class DynamicTriangulationThresholds(
    val correspondenceEpsilonMeters: Double,
    val forbiddenEpsilonMeters: Double,
    val minNormalizationConfidence: Float,
    val clusterRadiusMeters: Float,
    val conflictConfidenceThreshold: Float,
    val additionConfidenceThreshold: Float,
    val componentConfidenceThreshold: Float,
)

/**
 * Calculates gradually tightened triangulation thresholds for the accepted frame count.
 *
 * Thresholds stay at their initial values during the first five frames and change linearly until
 * reaching their strict values at thirty frames. Distance tolerances are halved, confidence
 * requirements move halfway towards one, and the component threshold grows by twenty percent.
 *
 * @param acceptedFrameCount number of accepted frames including the frame being processed.
 * @param correspondenceEpsilonMeters initial support-edge ray-distance threshold.
 * @param forbiddenEpsilonMeters initial forbidden-pair ray-distance threshold.
 * @param minNormalizationConfidence initial raw confidence mapped to zero.
 * @param maxNormalizationConfidence raw confidence mapped to one.
 * @param clusterRadiusMeters initial component cluster radius.
 * @param conflictConfidenceThreshold initial edge support threshold.
 * @param additionConfidenceThreshold initial observation-addition threshold.
 * @param componentConfidenceThreshold initial valid-component threshold.
 * @return thresholds tightened according to [acceptedFrameCount].
 */
internal fun dynamicTriangulationThresholds(
    acceptedFrameCount: Int,
    correspondenceEpsilonMeters: Double = 0.05,
    forbiddenEpsilonMeters: Double = 0.1,
    minNormalizationConfidence: Float = 0f,
    maxNormalizationConfidence: Float = 0.4f,
    clusterRadiusMeters: Float = 0.03f,
    conflictConfidenceThreshold: Float = 0.7f,
    additionConfidenceThreshold: Float = 0.5f,
    componentConfidenceThreshold: Float = 1.5f,
): DynamicTriangulationThresholds {
    require(acceptedFrameCount >= 0) {
        "acceptedFrameCount must not be negative"
    }
    require(forbiddenEpsilonMeters >= correspondenceEpsilonMeters) {
        "forbiddenEpsilonMeters must not be smaller than correspondenceEpsilonMeters"
    }
    require(maxNormalizationConfidence > minNormalizationConfidence) {
        "maxNormalizationConfidence must be greater than minNormalizationConfidence"
    }
    require(correspondenceEpsilonMeters >= 0.0 && clusterRadiusMeters >= 0f) {
        "Distance thresholds must not be negative"
    }

    val progress = (
        (acceptedFrameCount - THRESHOLD_TIGHTENING_START_FRAME).toFloat() /
            (THRESHOLD_TIGHTENING_END_FRAME - THRESHOLD_TIGHTENING_START_FRAME)
        ).coerceIn(0f, 1f)
    val distanceMultiplier = lerp(1f, STRICT_DISTANCE_MULTIPLIER, progress)
    val confidenceProgress = progress * STRICT_CONFIDENCE_PROGRESS
    val tightenedCorrespondenceEpsilon = correspondenceEpsilonMeters * distanceMultiplier
    val tightenedForbiddenEpsilon = forbiddenEpsilonMeters * distanceMultiplier

    return DynamicTriangulationThresholds(
        correspondenceEpsilonMeters = tightenedCorrespondenceEpsilon,
        forbiddenEpsilonMeters = maxOf(tightenedCorrespondenceEpsilon, tightenedForbiddenEpsilon),
        minNormalizationConfidence = lerp(
            minNormalizationConfidence,
            maxOf(minNormalizationConfidence, maxNormalizationConfidence * STRICT_MIN_NORMALIZATION_RATIO),
            progress,
        ),
        clusterRadiusMeters = clusterRadiusMeters * distanceMultiplier,
        conflictConfidenceThreshold = lerp(conflictConfidenceThreshold, 1f, confidenceProgress),
        additionConfidenceThreshold = lerp(additionConfidenceThreshold, 1f, confidenceProgress),
        componentConfidenceThreshold = componentConfidenceThreshold *
            lerp(1f, STRICT_COMPONENT_THRESHOLD_MULTIPLIER, progress),
    )
}

/**
 * Linearly interpolates between two floating-point values.
 *
 * @param start value returned at zero progress.
 * @param end value returned at full progress.
 * @param progress interpolation progress in the inclusive zero-to-one range.
 * @return interpolated value.
 */
private fun lerp(start: Float, end: Float, progress: Float): Float {
    return start + (end - start) * progress
}

private const val THRESHOLD_TIGHTENING_START_FRAME = 5
private const val THRESHOLD_TIGHTENING_END_FRAME = 30
private const val STRICT_DISTANCE_MULTIPLIER = 0.5f
private const val STRICT_CONFIDENCE_PROGRESS = 0.5f
private const val STRICT_MIN_NORMALIZATION_RATIO = 0.25f
private const val STRICT_COMPONENT_THRESHOLD_MULTIPLIER = 1.2f
