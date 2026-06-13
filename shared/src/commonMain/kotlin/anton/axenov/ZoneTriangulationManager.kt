package anton.axenov

import korlibs.math.geom.Vector3F as Vector3
import kotlin.random.Random

/**
 * Segmentation with all representative points derived from its instances.
 *
 * @param segmentationIndex insertion order index inside one zone manager.
 * @param frameSnapshot frame snapshot that produced the segmentation.
 * @param prediction segmentation result for the frame.
 * @param points points derived from segmentation instances.
 */
data class ZoneSegmentation(
    val segmentationIndex: Int,
    val zone: Zone,
    val frameSnapshot: DetectionFrameSnapshotDto,
    val prediction: SegmentationPrediction,
) {
    val points: List<ZoneTriangulationPoint> by lazy {
        filterWeakSegmentationInstances(prediction.instances).map { instance ->
            ZoneTriangulationPoint(
                segmentation = this,
                imagePoint = instance.bbox.centerPoint,
                confidence = instance.confidence,
            )
        }
    }
}

/**
 * Removes the weakest low-confidence observations before pairwise hypothesis generation.
 *
 * Only observations below confidence `0.15` are eligible for removal. At most the lowest ten
 * percent of all frame observations are removed, rounded down so filtering never exceeds ten
 * percent. Therefore the smaller restriction always wins.
 *
 * @param instances observations produced by one segmentation frame.
 * @return observations retained for ray matching and pairwise triangulation.
 */
internal fun filterWeakSegmentationInstances(
    instances: List<SegmentationInstance>,
): List<SegmentationInstance> {
    val maximumRemovalCount = instances.size / WEAK_OBSERVATION_FILTER_RATIO_DENOMINATOR
    if (maximumRemovalCount == 0) {
        return instances
    }
    val instancesToRemove = instances
        .withIndex()
        .filter { (_, instance) -> instance.confidence < WEAK_OBSERVATION_CONFIDENCE_THRESHOLD }
        .sortedWith(
            compareBy<IndexedValue<SegmentationInstance>> { (_, instance) -> instance.confidence }
                .thenBy { (index, _) -> index },
        )
        .take(maximumRemovalCount)
        .mapTo(mutableSetOf()) { (index, _) -> index }
    return instances.filterIndexed { index, _ -> index !in instancesToRemove }
}

/**
 * Representative point for one segmented object instance.
 *
 * @param segmentation source segmentation.
 * @param imagePoint representative image-space point for the instance.
 * @param confidence of the model result.
 */
data class ZoneTriangulationPoint(
    val segmentation: ZoneSegmentation,
    val imagePoint: ImagePoint,
    val confidence: Float,
) {
    /**
     * Sparse non-forbidden candidate sets keyed by segmentations that were actually compared.
     */
    val candidatesBySegmentation: MutableMap<ZoneSegmentation, Set<ZoneTriangulationPoint>> = mutableMapOf()

    /**
     * Segmentations for which this observation was excluded as too ambiguous.
     */
    val ignoredSegmentations: MutableSet<ZoneSegmentation> = mutableSetOf()

    /**
     * Checks whether this observation and [anotherPoint] were compared and rejected geometrically.
     *
     * @param anotherPoint observation from another segmentation.
     * @return true when the observations form an explicitly forbidden edge.
     */
    fun hasForbiddenEdgeTo(anotherPoint: ZoneTriangulationPoint): Boolean {
        if (
            anotherPoint.segmentation in ignoredSegmentations ||
            segmentation in anotherPoint.ignoredSegmentations
        ) {
            return false
        }
        val directCandidates = candidatesBySegmentation[anotherPoint.segmentation]
        if (directCandidates != null) {
            return anotherPoint !in directCandidates
        }
        val reverseCandidates = anotherPoint.candidatesBySegmentation[segmentation]
            ?: return false
        return this !in reverseCandidates
    }
}

/**
 * Manages triangulation candidates for all segmentation points of one zone.
 *
 * @param triangulationMath math helper used to find point correspondences between frames.
 * @param defaultEpsilonMeters default maximal physical distance between corresponding viewing rays.
 * @param defaultForbiddenEpsilonMeters distance after which a compared pair becomes explicitly forbidden.
 */
class ZoneTriangulationManager(
    private val triangulationMath: TriangulationMath = TriangulationMath(),
    private val defaultEpsilonMeters: Double = DEFAULT_CORRESPONDENCE_EPSILON_METERS,
    private val defaultForbiddenEpsilonMeters: Double = DEFAULT_FORBIDDEN_CORRESPONDENCE_EPSILON_METERS,
) {
    private val worldPointHypothesisResolversByZoneId: MutableMap<Long, WorldPointHypothesisResolver> = mutableMapOf()

    private val segmentationResults = mutableListOf<ZoneSegmentation>()

    /**
     * Returns resolved multi-view world points accumulated for this zone.
     *
     * @return immutable snapshot of resolved world points.
     */
    fun getResolvedWorldPoints(): List<WorldPoint> {
        val resolvedWorldPoints = mutableListOf<WorldPoint>()

        worldPointHypothesisResolversByZoneId.forEach { (_, resolver) ->
            resolvedWorldPoints.addAll(resolver.resolvedComponents.map { it.toWorldPoint() })
        }
        return resolvedWorldPoints.toList()
    }

    /**
     * Adds one segmentation result for the zone and creates candidate sets for all of its points.
     *
     * @param frameSnapshot frame snapshot that produced the segmentation.
     * @param prediction segmentation result for the frame.
     * @param epsilonMeters explicit maximal ray distance, or null to use the dynamic threshold.
     * @param forbiddenEpsilonMeters explicit forbidden-pair distance, or null to use the dynamic threshold.
     * @return true when the frame passed consistency checks and was accepted.
     */
    fun addSegmentationResult(
        zone: Zone,
        frameSnapshot: DetectionFrameSnapshotDto,
        prediction: SegmentationPrediction,
        epsilonMeters: Double? = null,
        forbiddenEpsilonMeters: Double? = null,
    ): Boolean {
        val acceptedFrameCount = segmentationResults.size + 1
        val dynamicThresholds = dynamicTriangulationThresholds(
            acceptedFrameCount = acceptedFrameCount,
            correspondenceEpsilonMeters = defaultEpsilonMeters,
            forbiddenEpsilonMeters = defaultForbiddenEpsilonMeters,
        )
        val effectiveEpsilonMeters = epsilonMeters ?: dynamicThresholds.correspondenceEpsilonMeters
        val effectiveForbiddenEpsilonMeters = forbiddenEpsilonMeters
            ?: maxOf(dynamicThresholds.forbiddenEpsilonMeters, effectiveEpsilonMeters)
        val newPoints = mutableListOf<WorldPoint>()
        val newSegmentation = ZoneSegmentation(
            segmentationIndex = segmentationResults.size,
            zone = zone,
            frameSnapshot = frameSnapshot,
            prediction = prediction,
        )
        val resolver = worldPointHypothesisResolversByZoneId.getOrPut(zone.id) { WorldPointHypothesisResolver() }
        val candidateComponentsByObservation = findCandidateComponentsByObservationRay(
            segmentation = newSegmentation,
            components = resolver.resolvedComponents,
            maximumDistanceMeters = dynamicThresholds.clusterRadiusMeters,
        )
        val segmentationsToCompare = selectSegmentationsForComparison(
            newSnapshot = frameSnapshot,
            previousSegmentations = segmentationResults,
        )
        for (segmentation in segmentationsToCompare) {
            newPoints += getPointsCandidatesBySegmentation(
                segmentation = newSegmentation,
                anotherSegmentation = segmentation,
                epsilonMeters = effectiveEpsilonMeters,
                forbiddenEpsilonMeters = effectiveForbiddenEpsilonMeters,
            )
        }
        val supportedObservationCount = newPoints
            .flatMap { worldPoint -> worldPoint.parentPoints }
            .filter { point -> point.segmentation === newSegmentation }
            .toSet()
            .size
        if (
            shouldRejectFrameByUnsupportedObservations(
                acceptedFrameCount = segmentationResults.size,
                observationCount = newSegmentation.points.size,
                supportedObservationCount = supportedObservationCount,
            )
        ) {
            removeSegmentationRelationships(newSegmentation)
            return false
        }

        segmentationResults += newSegmentation
        resolver.resolve(
            newWorldPoints = newPoints,
            acceptedFrameCount = acceptedFrameCount,
            candidateComponentsByObservation = candidateComponentsByObservation,
        )
        return true
    }

    /**
     * Removes temporary candidate and ignore-state relationships to a rejected segmentation.
     *
     * @param rejectedSegmentation segmentation that must not remain referenced by accepted frames.
     */
    private fun removeSegmentationRelationships(rejectedSegmentation: ZoneSegmentation) {
        segmentationResults.forEach { segmentation ->
            segmentation.points.forEach { point ->
                point.candidatesBySegmentation.remove(rejectedSegmentation)
                point.ignoredSegmentations.remove(rejectedSegmentation)
            }
        }
    }

    /**
     * Builds candidate groups between two segmentations.
     *
     * @param epsilonMeters maximal physical distance between corresponding viewing rays.
     * @param forbiddenEpsilonMeters distance after which a compared pair becomes explicitly forbidden.
     * @return map of older segmentation index to all candidate points from that segmentation.
     */
    private fun getPointsCandidatesBySegmentation(
        segmentation: ZoneSegmentation,
        anotherSegmentation: ZoneSegmentation,
        epsilonMeters: Double,
        forbiddenEpsilonMeters: Double,
    ): List<WorldPoint> {
        val newPoints = mutableListOf<WorldPoint>()
        val correspondence = triangulationMath.correspondenceCandidates(
            firstSnapshot = segmentation.frameSnapshot,
            secondSnapshot = anotherSegmentation.frameSnapshot,
            firstImagePoints = segmentation.points.map { it.imagePoint },
            secondImagePoints = anotherSegmentation.points.map { it.imagePoint },
            epsilonMeters = epsilonMeters,
            maxDistanceMeters = forbiddenEpsilonMeters,
        )
        val ambiguousIndices = findAmbiguousCorrespondenceIndices(
            correspondence = correspondence,
            firstObservationCount = segmentation.points.size,
            secondObservationCount = anotherSegmentation.points.size,
            epsilonMeters = epsilonMeters,
        )
        ambiguousIndices.first.forEach { index ->
            segmentation.points[index].ignoredSegmentations += anotherSegmentation
        }
        ambiguousIndices.second.forEach { index ->
            anotherSegmentation.points[index].ignoredSegmentations += segmentation
        }
        val reverseCandidatesByPoint = anotherSegmentation.points
            .associateWith { mutableSetOf<ZoneTriangulationPoint>() }
        segmentation.points.forEachIndexed { index, point ->
            if (index in ambiguousIndices.first) {
                return@forEachIndexed
            }
            val candidatePoints = correspondence[index].map { (index, midPoint) ->
                Pair(anotherSegmentation.points[index], midPoint)
            }.filterNot { (candidatePoint, _) ->
                candidatePoint.segmentation in point.ignoredSegmentations ||
                    point.segmentation in candidatePoint.ignoredSegmentations
            }
            val candidateObservations = candidatePoints.map { it.first }.toSet()
            point.candidatesBySegmentation[anotherSegmentation] = candidateObservations
            candidateObservations.forEach { candidateObservation ->
                reverseCandidatesByPoint.getValue(candidateObservation) += point
            }
            newPoints += getWorldPointsForCandidates(
                point = point,
                candidatePoints = candidatePoints,
                epsilonMeters = epsilonMeters,
            )
        }
        reverseCandidatesByPoint.forEach { (point, reverseCandidates) ->
            if (segmentation !in point.ignoredSegmentations) {
                point.candidatesBySegmentation[segmentation] = reverseCandidates
            }
        }
        return newPoints
    }

    /**
     * Triangulates all candidate pairs for one newly added point.
     *
     * @param point point from the newly added segmentation.
     * @param candidatePoints candidate corresponding points from another segmentation.
     * @param epsilonMeters maximal physical distance for a supporting edge.
     */
    private fun getWorldPointsForCandidates(
        point: ZoneTriangulationPoint,
        candidatePoints: List<Pair<ZoneTriangulationPoint, TriangulationMath.RaysMidPoint>>,
        epsilonMeters: Double,
    ): List<WorldPoint> {
        return candidatePoints.mapNotNull { candidatePoint ->
            val worldPosition = candidatePoint.second.midPoint

            // Keep neutral pairs up to epsilon_max in candidate storage, but do not create support edges.
            if (
                candidatePoint.second.distance > epsilonMeters ||
                !point.segmentation.zone.boundingBox.containsPoint(worldPosition)
            )
                null
            else
                WorldPoint(
                    position = worldPosition,
                    parentPoints = setOf(point, candidatePoint.first),
                    confidence = point.confidence *
                            candidatePoint.first.confidence *
                            candidatePoint.second.distanceConfidence *
                            candidatePoint.second.angleConfidence,
                )
        }
    }

}

/**
 * Finds existing components close to each new observation's viewing ray.
 *
 * This search is intentionally performed before any pairwise [WorldPoint] is generated for the
 * new frame. Components are returned in ascending ray-distance order so the resolver tries the
 * geometrically most plausible component first.
 *
 * @param segmentation newly received segmentation whose observations are being processed.
 * @param components existing resolved components eligible for direct observation inclusion.
 * @param maximumDistanceMeters maximal distance from component center to observation ray.
 * @return candidate components by new observation, each ordered by ascending ray distance.
 */
internal fun findCandidateComponentsByObservationRay(
    segmentation: ZoneSegmentation,
    components: List<WorldPointHypothesisResolver.HypothesisComponent>,
    maximumDistanceMeters: Float,
    triangulationMath: TriangulationMath = TriangulationMath(),
): Map<ZoneTriangulationPoint, List<WorldPointHypothesisResolver.HypothesisComponent>> {
    require(maximumDistanceMeters >= 0f) {
        "maximumDistanceMeters must not be negative"
    }
    return segmentation.points.associateWith { observation ->
        components
            .asSequence()
            .filterNot { component -> observation.segmentation in component.selectedByFrame }
            .map { component ->
                component to triangulationMath.distanceToViewingRay(
                    frame = segmentation.frameSnapshot,
                    imagePoint = observation.imagePoint,
                    worldPoint = component.center,
                )
            }
            .filter { (_, distance) -> distance <= maximumDistanceMeters }
            .sortedBy { (_, distance) -> distance }
            .map { (component, _) -> component }
            .toList()
    }
}

/**
 * Checks whether a frame has too many observations without any supporting triangulation edge.
 *
 * The first two accepted frames form the warm-up period and are never rejected by this check.
 * Starting with the third frame, a frame is rejected when more than 30% of its observations did
 * not produce any supporting edge to selected previous frames.
 *
 * @param acceptedFrameCount number of already accepted frames.
 * @param observationCount total observations on the candidate frame.
 * @param supportedObservationCount observations that produced at least one supporting edge.
 * @return true when the whole candidate frame must be rejected.
 */
internal fun shouldRejectFrameByUnsupportedObservations(
    acceptedFrameCount: Int,
    observationCount: Int,
    supportedObservationCount: Int,
): Boolean {
    require(acceptedFrameCount >= 0) {
        "acceptedFrameCount must not be negative"
    }
    require(observationCount >= 0) {
        "observationCount must not be negative"
    }
    require(supportedObservationCount in 0..observationCount) {
        "supportedObservationCount must be within observationCount"
    }
    if (acceptedFrameCount < FRAME_CONSISTENCY_WARM_UP_COUNT) {
        return false
    }
    if (observationCount == 0) {
        return true
    }
    val unsupportedObservationCount = observationCount - supportedObservationCount
    return unsupportedObservationCount.toLong() * UNSUPPORTED_RATIO_DENOMINATOR >
        observationCount.toLong() * UNSUPPORTED_RATIO_NUMERATOR
}

/**
 * Sorts previous segmentations by frame-pair quality and applies probabilistic comparison thinning.
 *
 * The ten most different frames are always used. Next 10 ranked frames are used with
 * probability 0.5, all lower-ranked frames are used with probability 0.25.
 *
 * @param newSnapshot snapshot that is about to be triangulated.
 * @param previousSegmentations accumulated segmentations eligible for comparison.
 * @param random random source used for probabilistic thinning.
 * @return selected segmentations ordered by descending frame difference quality.
 */
internal fun selectSegmentationsForComparison(
    newSnapshot: DetectionFrameSnapshotDto,
    previousSegmentations: List<ZoneSegmentation>,
    random: Random = Random.Default,
): List<ZoneSegmentation> {
    return previousSegmentations
        .sortedWith(
            compareByDescending<ZoneSegmentation> { segmentation ->
                frameDifferenceQuality(newSnapshot, segmentation.frameSnapshot)
            }.thenBy { segmentation -> segmentation.segmentationIndex },
        )
        .filterIndexed { index, _ -> shouldKeepFrameComparisonRank(index, random) }
}

/**
 * Calculates usefulness of comparing two frames from camera baseline and view-direction difference.
 *
 * @param firstSnapshot first frame snapshot.
 * @param secondSnapshot second frame snapshot.
 * @return non-negative frame difference quality score.
 */
internal fun frameDifferenceQuality(
    firstSnapshot: DetectionFrameSnapshotDto,
    secondSnapshot: DetectionFrameSnapshotDto,
): Float {
    val positionDifference = (
        firstSnapshot.cameraPose.translation -
            secondSnapshot.cameraPose.translation
        ).length
    val firstForward = cameraForwardDirection(firstSnapshot)
    val secondForward = cameraForwardDirection(secondSnapshot)
    val directionDifference = ((1f - firstForward.dot(secondForward).coerceIn(-1f, 1f)) * 0.5f)
    return positionDifference + directionDifference
}

/**
 * Randomly decides whether a frame comparison at one zero-based quality rank should be retained.
 *
 * @param rank zero-based rank after sorting by descending frame difference quality.
 * @param random random source used for the decision.
 * @return true when the ranked frame should be compared.
 */
internal fun shouldKeepFrameComparisonRank(
    rank: Int,
    random: Random = Random.Default,
): Boolean {
    require(rank >= 0) {
        "rank must not be negative"
    }
    return when {
        rank < ALWAYS_COMPARE_FRAME_COUNT -> true
        rank < HALF_COMPARE_END_RANK -> random.nextFloat() < HALF_COMPARE_PROBABILITY
        else -> random.nextFloat() < LATER_COMPARE_PROBABILITY
    }
}

/**
 * Returns normalized world-space camera forward direction for one snapshot.
 *
 * @param snapshot frame snapshot with camera rotation.
 * @return normalized world-space forward direction.
 */
private fun cameraForwardDirection(snapshot: DetectionFrameSnapshotDto): Vector3 {
    return snapshot.cameraPose.rotationQuaternion
        .normalized()
        .transform(Vector3(0f, 0f, -1f))
        .normalized()
}

/**
 * Finds observations that have too many supporting epipolar correspondences in either frame.
 *
 * @param correspondence non-forbidden candidates from first-frame observations to second-frame observations.
 * @param firstObservationCount number of observations on the first frame.
 * @param secondObservationCount number of observations on the second frame.
 * @param epsilonMeters maximal ray distance for a candidate to count as a supporting correspondence.
 * @return indices of ambiguous observations on the first and second frames.
 */
internal fun findAmbiguousCorrespondenceIndices(
    correspondence: List<List<Pair<Int, TriangulationMath.RaysMidPoint>>>,
    firstObservationCount: Int,
    secondObservationCount: Int,
    epsilonMeters: Double,
): Pair<Set<Int>, Set<Int>> {
    require(correspondence.size == firstObservationCount) {
        "Correspondence list size must match firstObservationCount"
    }
    val firstLimit = maximumEpipolarCorrespondenceCount(secondObservationCount)
    val secondLimit = maximumEpipolarCorrespondenceCount(firstObservationCount)
    val secondSupportCounts = IntArray(secondObservationCount)
    val ambiguousFirstIndices = mutableSetOf<Int>()

    correspondence.forEachIndexed { firstIndex, candidates ->
        val supportingCandidates = candidates.filter { (_, midpoint) ->
            midpoint.distance <= epsilonMeters
        }
        if (supportingCandidates.size > firstLimit) {
            ambiguousFirstIndices += firstIndex
        }
        supportingCandidates.forEach { (secondIndex, _) ->
            require(secondIndex in secondSupportCounts.indices) {
                "Correspondence candidate index must reference the second frame"
            }
            secondSupportCounts[secondIndex]++
        }
    }

    val ambiguousSecondIndices = secondSupportCounts.indices
        .filterTo(mutableSetOf()) { index -> secondSupportCounts[index] > secondLimit }
    return ambiguousFirstIndices to ambiguousSecondIndices
}

/**
 * Calculates the maximal useful correspondence count for one observation.
 *
 * @param oppositeFrameObservationCount number of observations on the opposite frame.
 * @return inclusive upper bound `ceil(sqrt(oppositeFrameObservationCount))`.
 */
internal fun maximumEpipolarCorrespondenceCount(oppositeFrameObservationCount: Int): Int {
    require(oppositeFrameObservationCount >= 0) {
        "oppositeFrameObservationCount must not be negative"
    }
    return kotlin.math.ceil(kotlin.math.sqrt(oppositeFrameObservationCount.toDouble())).toInt()
}

private const val DEFAULT_CORRESPONDENCE_EPSILON_METERS = 0.05
private const val DEFAULT_FORBIDDEN_CORRESPONDENCE_EPSILON_METERS = 0.1
private const val ALWAYS_COMPARE_FRAME_COUNT = 10
private const val HALF_COMPARE_END_RANK = 20
private const val HALF_COMPARE_PROBABILITY = 0.5f
private const val LATER_COMPARE_PROBABILITY = 0.25f
private const val FRAME_CONSISTENCY_WARM_UP_COUNT = 3
private const val UNSUPPORTED_RATIO_NUMERATOR = 3L
private const val UNSUPPORTED_RATIO_DENOMINATOR = 10L
private const val WEAK_OBSERVATION_CONFIDENCE_THRESHOLD = 0.15f
private const val WEAK_OBSERVATION_FILTER_RATIO_DENOMINATOR = 10
