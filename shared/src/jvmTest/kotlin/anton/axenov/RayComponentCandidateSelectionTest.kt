package anton.axenov

import kotlin.test.Test
import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F

/**
 * Tests for component selection by distance to a new observation's viewing ray.
 */
class RayComponentCandidateSelectionTest {
    @Test
    fun `distanceToViewingRay should measure forward ray rather than infinite line`() {
        val math = TriangulationMath()
        val snapshot = rayCandidateSnapshot()

        math.distanceToViewingRay(snapshot, ImagePoint(0, 0), Vector3F(0.2f, 0f, -5f)) shouldBe
            (0.2f plusOrMinus 0.0001f)
        math.distanceToViewingRay(snapshot, ImagePoint(0, 0), Vector3F(0f, 0f, 2f)) shouldBe
            (2f plusOrMinus 0.0001f)
    }

    @Test
    fun `candidate components should be filtered and ordered by observation ray before edge generation`() {
        val resolver = WorldPointHypothesisResolver()
        val segmentation = rayCandidateSegmentation()
        val closest = rayCandidateComponent(resolver, center = Vector3F(0.01f, 0f, -5f), id = 10)
        val second = rayCandidateComponent(resolver, center = Vector3F(0.02f, 0f, -5f), id = 20)
        val outside = rayCandidateComponent(resolver, center = Vector3F(0.2f, 0f, -5f), id = 30)

        val candidates = findCandidateComponentsByObservationRay(
            segmentation = segmentation,
            components = listOf(outside, second, closest),
            maximumDistanceMeters = 0.05f,
        )

        candidates.size shouldBe 1
        candidates.getValue(segmentation.points.single()) shouldBe listOf(closest, second)
    }
}

/**
 * Creates one component centered at a controlled world-space position.
 *
 * @param resolver resolver owning the component.
 * @param center desired component center.
 * @param id stable synthetic identifier.
 * @return synthetic component.
 */
private fun rayCandidateComponent(
    resolver: WorldPointHypothesisResolver,
    center: Vector3F,
    id: Int,
): WorldPointHypothesisResolver.HypothesisComponent {
    val first = rayCandidateObservation(id)
    val second = rayCandidateObservation(id + 1)
    return resolver.HypothesisComponent(
        WorldPoint(
            position = center,
            parentPoints = setOf(first, second),
            confidence = 1f,
        ),
    )
}

/**
 * Creates one synthetic observation on its own frame.
 *
 * @param id stable synthetic identifier.
 * @return synthetic observation.
 */
private fun rayCandidateObservation(id: Int): ZoneTriangulationPoint {
    return ZoneTriangulationPoint(
        segmentation = rayCandidateSegmentation(index = id, withObservation = false),
        imagePoint = ImagePoint(0, 0),
        confidence = 1f,
    )
}

/**
 * Creates a segmentation with one center-pixel observation by default.
 *
 * @param index stable segmentation index.
 * @param withObservation whether to include one predicted observation.
 * @return synthetic segmentation.
 */
private fun rayCandidateSegmentation(
    index: Int = 0,
    withObservation: Boolean = true,
): ZoneSegmentation {
    val instances = if (withObservation) {
        listOf(
            SegmentationInstance(
                id = 1,
                bbox = SegmentationBoundingBox(x = 0, y = 0, width = 0, height = 0),
                polygon = listOf(ImagePoint(0, 0), ImagePoint(1, 0), ImagePoint(0, 1)),
                confidence = 1f,
            ),
        )
    } else {
        emptyList()
    }
    return ZoneSegmentation(
        segmentationIndex = index,
        zone = rayCandidateZone(),
        frameSnapshot = rayCandidateSnapshot(),
        prediction = SegmentationPrediction(
            filename = "$index.png",
            width = 1,
            height = 1,
            count = instances.size,
            instances = instances,
        ),
    )
}

/**
 * Creates a camera at the origin looking along the negative Z axis.
 *
 * @return synthetic frame snapshot.
 */
private fun rayCandidateSnapshot(): DetectionFrameSnapshotDto {
    return DetectionFrameSnapshotDto(
        screenshotPngBytes = byteArrayOf(),
        frameTimestamp = 0L,
        imageWidth = 1,
        imageHeight = 1,
        focalLengthX = 1f,
        focalLengthY = 1f,
        principalPointX = 0f,
        principalPointY = 0f,
        distortionCoefficients = emptyList(),
        cameraPose = CameraPoseDto(
            translation = Vector3F(0f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
        ),
        depthSnapshot = null,
    )
}

/**
 * Creates one synthetic zone containing the candidate component centers.
 *
 * @return synthetic zone.
 */
private fun rayCandidateZone(): Zone {
    return Zone(
        id = 1L,
        sampledPoints = listOf(Vector3F(0f, 0f, -5f)),
        planePose = PlanePose(
            center = Vector3F(0f, 0f, -5f),
            rotation = Quaternion.IDENTITY,
            normal = Vector3F(0f, 1f, 0f),
        ),
    )
}
