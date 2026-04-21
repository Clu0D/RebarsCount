package anton.axenov

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
        prediction.instances.map { instance ->
            ZoneTriangulationPoint(
                segmentation = this,
                imagePoint = instance.bbox.centerPoint,
                confidence = instance.confidence,
            )
        }
    }
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
    val candidatesBySegmentation: MutableMap<ZoneSegmentation, Set<ZoneTriangulationPoint>> = mutableMapOf()
}

/**
 * Manages triangulation candidates for all segmentation points of one zone.
 *
 * @param triangulationMath math helper used to find point correspondences between frames.
 * @param defaultEpsilonMeters default maximal physical distance between corresponding viewing rays.
 */
class ZoneTriangulationManager(
    private val triangulationMath: TriangulationMath = TriangulationMath(),
    private val defaultEpsilonMeters: Double = DEFAULT_CORRESPONDENCE_EPSILON_METERS,
) {

    private val segmentationResults = mutableListOf<ZoneSegmentation>()
    private val worldPoints = mutableListOf<WorldPoint>()

    /**
     * Returns all reconstructed world points accumulated for this zone.
     *
     * @return immutable snapshot of world points.
     */
    fun getWorldPoints(): List<WorldPoint> = worldPoints.toList()

    /**
     * Adds one segmentation result for the zone and creates candidate sets for all of its points.
     *
     * @param frameSnapshot frame snapshot that produced the segmentation.
     * @param prediction segmentation result for the frame.
     * @param epsilonMeters maximal physical distance between corresponding viewing rays.
     * @return candidate sets created for points of the newly added segmentation.
     */
    fun addSegmentationResult(
        zone: Zone,
        frameSnapshot: DetectionFrameSnapshotDto,
        prediction: SegmentationPrediction,
        epsilonMeters: Double = defaultEpsilonMeters,
    ) {
        val newSegmentation = ZoneSegmentation(
            segmentationIndex = segmentationResults.size,
            zone = zone,
            frameSnapshot = frameSnapshot,
            prediction = prediction,
        )
        for (segmentation in segmentationResults)
            addPointsCandidatesBySegmentation(newSegmentation, segmentation, epsilonMeters)

        segmentationResults += newSegmentation
    }

    /**
     * Builds candidate groups between two segmentations.
     *
     * @param epsilonMeters maximal physical distance between corresponding viewing rays.
     * @return map of older segmentation index to all candidate points from that segmentation.
     */
    private fun addPointsCandidatesBySegmentation(
        segmentation: ZoneSegmentation,
        anotherSegmentation: ZoneSegmentation,
        epsilonMeters: Double
    ) {
        val correspondence = triangulationMath.correspondenceCandidates(
            firstSnapshot = segmentation.frameSnapshot,
            secondSnapshot = anotherSegmentation.frameSnapshot,
            firstImagePoints = segmentation.points.map { it.imagePoint },
            secondImagePoints = anotherSegmentation.points.map { it.imagePoint },
            epsilonMeters = epsilonMeters,
        )
        segmentation.points.forEachIndexed { index, point ->
            val candidatePoints = correspondence[index].map { (index, midPoint) ->
                Pair(anotherSegmentation.points[index], midPoint)
            }
            point.candidatesBySegmentation[anotherSegmentation] = candidatePoints.map { it.first }.toSet()
            addWorldPointsForCandidates(point, candidatePoints)
        }
    }

    /**
     * Triangulates all candidate pairs for one newly added point.
     *
     * @param point point from the newly added segmentation.
     * @param candidatePoints candidate corresponding points from another segmentation.
     */
    private fun addWorldPointsForCandidates(
        point: ZoneTriangulationPoint,
        candidatePoints: List<Pair<ZoneTriangulationPoint, TriangulationMath.RaysMidPoint>>,
    ) {
        candidatePoints.forEach { candidatePoint ->
            val worldPosition = candidatePoint.second.midPoint

            // filter point out if not inside zone bounding box
            if (!point.segmentation.zone.boundingBox.containsPoint(worldPosition)) {
                return@forEach
            }

            worldPoints += WorldPoint(
                position = worldPosition,
                parentPoints = setOf(point, candidatePoint.first),
                confidence = listOf(point.confidence, candidatePoint.first.confidence).average().toFloat(),
            )
        }
    }
}

private const val DEFAULT_CORRESPONDENCE_EPSILON_METERS = 0.05
