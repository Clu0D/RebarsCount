package anton.axenov

import korlibs.math.geom.Vector3F

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
    val frameSnapshot: DetectionFrameSnapshotDto,
    val prediction: SegmentationPrediction,
) {
    val points: List<ZoneTriangulationPoint> by lazy {
        prediction.instances.map { instance ->
            ZoneTriangulationPoint(
                segmentation = this,
                imagePoint = instance.bbox.centerPoint,
            )
        }
    }
}

/**
 * Stores one representative point for one segmented object instance.
 *
 * @param segmentation source segmentation.
 * @param imagePoint representative image-space point for the instance.
 */
data class ZoneTriangulationPoint(
    val segmentation: ZoneSegmentation,
    val imagePoint: ImagePoint,
) {
    val candidatesBySegmentation: MutableMap<ZoneSegmentation, Set<ZoneTriangulationPoint>> = mutableMapOf()
}

/**
 * Triangulated world-space point derived from corresponding 2D segmentation points.
 *
 * @param position reconstructed 3D point in world coordinates.
 * @param parentPoints source 2D points that were used to reconstruct this world point.
 */
data class WorldPoint(
    val position: Vector3F,
    val parentPoints: Set<ZoneTriangulationPoint>,
)

/**
 * Manages triangulation candidates for all segmentation points of one zone.
 *
 * @param triangulationMath math helper used to find point correspondences between frames.
 * @param defaultEpsilonPx default maximal epipolar distance in pixels.
 */
class ZoneTriangulationManager(
    private val triangulationMath: TriangulationMath = TriangulationMath(),
    private val defaultEpsilonPx: Double = DEFAULT_EPIPOLAR_EPSILON_PX,
) {

    private val segmentationResults = mutableListOf<ZoneSegmentation>()
    val worldPoints = mutableListOf<WorldPoint>()

    /**
     * Adds one segmentation result for the zone and creates candidate sets for all of its points.
     *
     * @param frameSnapshot frame snapshot that produced the segmentation.
     * @param prediction segmentation result for the frame.
     * @param epsilonPx maximal distance in pixels from the epipolar line.
     * @return candidate sets created for points of the newly added segmentation.
     */
    fun addSegmentationResult(
        frameSnapshot: DetectionFrameSnapshotDto,
        prediction: SegmentationPrediction,
        epsilonPx: Double = defaultEpsilonPx,
    ) {
        val newSegmentation = ZoneSegmentation(
            segmentationIndex = segmentationResults.size,
            frameSnapshot = frameSnapshot,
            prediction = prediction,
        )
        for (segmentation in segmentationResults)
            addPointsCandidatesBySegmentation(newSegmentation, segmentation, epsilonPx)

        segmentationResults += newSegmentation
    }

    /**
     * Builds candidate groups between two segmentations.
     *
     * @param epsilonPx maximal distance in pixels from the epipolar line.
     * @return map of older segmentation index to all candidate points from that segmentation.
     */
    private fun addPointsCandidatesBySegmentation(
        segmentation: ZoneSegmentation,
        anotherSegmentation: ZoneSegmentation,
        epsilonPx: Double
    ) {
        val correspondence = triangulationMath.correspondenceCandidates(
            firstSnapshot = segmentation.frameSnapshot,
            secondSnapshot = anotherSegmentation.frameSnapshot,
            firstImagePoints = segmentation.points.map { it.imagePoint },
            secondImagePoints = anotherSegmentation.points.map { it.imagePoint },
            epsilonPx = epsilonPx,
        )
        segmentation.points.forEachIndexed { index, point ->
            val candidatePoints = correspondence[index].map { anotherSegmentation.points[it] }.toSet()
            point.candidatesBySegmentation[anotherSegmentation] = candidatePoints
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
        candidatePoints: Set<ZoneTriangulationPoint>,
    ) {
        candidatePoints.forEach { candidatePoint ->
            val worldPosition = triangulationMath.triangulateWorldPoint(
                firstSnapshot = point.segmentation.frameSnapshot,
                secondSnapshot = candidatePoint.segmentation.frameSnapshot,
                firstImagePoint = point.imagePoint,
                secondImagePoint = candidatePoint.imagePoint,
            ) ?: return@forEach
            worldPoints += WorldPoint(
                position = worldPosition,
                parentPoints = setOf(point, candidatePoint),
            )
        }
    }
}

private const val DEFAULT_EPIPOLAR_EPSILON_PX = 5.0
