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
     * @param epsilonMeters maximal physical distance between corresponding viewing rays.
     * @return candidate sets created for points of the newly added segmentation.
     */
    fun addSegmentationResult(
        zone: Zone,
        frameSnapshot: DetectionFrameSnapshotDto,
        prediction: SegmentationPrediction,
        epsilonMeters: Double = defaultEpsilonMeters,
    ) {
        val newPoints = mutableListOf<WorldPoint>()
        val newSegmentation = ZoneSegmentation(
            segmentationIndex = segmentationResults.size,
            zone = zone,
            frameSnapshot = frameSnapshot,
            prediction = prediction,
        )
        for (segmentation in segmentationResults)
            newPoints += getPointsCandidatesBySegmentation(newSegmentation, segmentation, epsilonMeters)

        segmentationResults += newSegmentation
        resolveWorldPoints(newSegmentation.zone, newPoints)
    }

    /**
     * Builds candidate groups between two segmentations.
     *
     * @param epsilonMeters maximal physical distance between corresponding viewing rays.
     * @return map of older segmentation index to all candidate points from that segmentation.
     */
    private fun getPointsCandidatesBySegmentation(
        segmentation: ZoneSegmentation,
        anotherSegmentation: ZoneSegmentation,
        epsilonMeters: Double
    ): List<WorldPoint> {
        val newPoints = mutableListOf<WorldPoint>()
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
            newPoints += getWorldPointsForCandidates(point, candidatePoints)
        }
        return newPoints
    }

    /**
     * Triangulates all candidate pairs for one newly added point.
     *
     * @param point point from the newly added segmentation.
     * @param candidatePoints candidate corresponding points from another segmentation.
     */
    private fun getWorldPointsForCandidates(
        point: ZoneTriangulationPoint,
        candidatePoints: List<Pair<ZoneTriangulationPoint, TriangulationMath.RaysMidPoint>>,
    ): List<WorldPoint> {
        return candidatePoints.mapNotNull { candidatePoint ->
            val worldPosition = candidatePoint.second.midPoint

            // filter point out if not inside zone bounding box
            if (!point.segmentation.zone.boundingBox.containsPoint(worldPosition))
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

    /**
     * Rebuilds resolved multi-view point centers from all currently accumulated 2-view WorldPoints.
     */
    private fun resolveWorldPoints(zone: Zone, newPoints: MutableList<WorldPoint>) {
        val resolver = worldPointHypothesisResolversByZoneId.getOrPut(zone.id) { WorldPointHypothesisResolver() }
        resolver.resolve(newPoints)
    }
}

private const val DEFAULT_CORRESPONDENCE_EPSILON_METERS = 0.05
