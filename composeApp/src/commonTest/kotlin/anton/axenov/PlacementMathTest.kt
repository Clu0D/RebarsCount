package anton.axenov

import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import kotlin.random.Random
import kotlin.test.Test
import korlibs.math.geom.Vector3F as Vector3

/**
 * Placement math tests for common multiplatform logic.
 */
class PlacementMathTest {
    @Test
    fun `median candidate selection returns null on empty list`() {
        selectMedianPointCandidate(emptyList()) shouldBe null
    }

    @Test
    fun `median candidate selection returns works on sample case`() {
        val candidates = listOf(
            DepthCandidate(9, 9, 1000),
            DepthCandidate(1, 3, 950),
            DepthCandidate(1, 2, 900),
            DepthCandidate(3, 1, 975),
            DepthCandidate(2, 7, 990),
        )

        selectMedianPointCandidate(candidates) shouldBe DepthCandidate(2, 7, 990)
    }

    @Test
    fun `image-point sampling handles invalid sizes and clamped randomized bounds`() {
        val invalid = sampleImagePointsInBoundingBox(
            boundingBox = BoundingBox(0, 0, 10, 10),
            imageWidth = 0,
            imageHeight = 20,
            count = 4,
            random = Random(1),
        )
        invalid shouldBe emptyList()

        val points = sampleImagePointsInBoundingBox(
            boundingBox = BoundingBox(left = -10, top = -5, right = 8, bottom = 6),
            imageWidth = 9,
            imageHeight = 7,
            count = 3,
            random = Random(42),
        )

        points.size shouldBe 3
        points.first() shouldBe ImagePoint(4, 3)
        points.all { it.x in 0..8 && it.y in 0..6 } shouldBe true
    }

    @Test
    fun `image-point sampling still returns center when count is zero`() {
        val points = sampleImagePointsInBoundingBox(
            boundingBox = BoundingBox(left = 3, top = 3, right = 3, bottom = 3),
            imageWidth = 10,
            imageHeight = 10,
            count = 0,
            random = Random(0),
        )

        points shouldBe listOf(ImagePoint(3, 3))
    }

    @Test
    fun `image-to-view mapping handles normal and invalid dimensions`() {
        val mapped = mapImagePointsToViewPoints(
            imagePoints = listOf(ImagePoint(50, 25), ImagePoint(0, 0)),
            imageWidth = 100,
            imageHeight = 50,
            viewWidth = 200,
            viewHeight = 100,
        )

        mapped shouldBe listOf(
            ViewPoint(100f, 50f),
            ViewPoint(0f, 0f),
        )

        mapImagePointsToViewPoints(
            imagePoints = listOf(ImagePoint(1, 1)),
            imageWidth = 100,
            imageHeight = 100,
            viewWidth = -1,
            viewHeight = 100,
        ) shouldBe emptyList()
    }

    @Test
    fun `physical rectangle size uses fallback depth and clamps output`() {
        val fromDefaultDepth = computeRectanglePhysicalSize(
            boundingBox = BoundingBox(0, 0, 100, 50),
            depthMeters = null,
            focalLengthX = 1000f,
            focalLengthY = 500f,
            minRectangleSizeMeters = 0.1f,
            maxRectangleSizeMeters = 1.0f,
            minDepthMeters = 0.2f,
            maxDepthMeters = 4f,
            defaultDepthMeters = 2f,
        )
        fromDefaultDepth.first shouldBe (0.2f plusOrMinus 0.0001f)
        fromDefaultDepth.second shouldBe (0.2f plusOrMinus 0.0001f)

        val clamped = computeRectanglePhysicalSize(
            boundingBox = BoundingBox(5, 5, 3, 4),
            depthMeters = 100f,
            focalLengthX = 100f,
            focalLengthY = 100f,
            minRectangleSizeMeters = 0.3f,
            maxRectangleSizeMeters = 2.0f,
            minDepthMeters = 0.5f,
            maxDepthMeters = 5f,
            defaultDepthMeters = 1f,
        )
        clamped.first shouldBe (0.3f plusOrMinus 0.0001f)
        clamped.second shouldBe (0.3f plusOrMinus 0.0001f)
    }

    @Test
    fun `plane fit returns explicit error when points are insufficient`() {
        val result = fitPlanePoseFromPoints(
            worldPoints = listOf(Vector3(0f, 0f, 0f)),
            cameraPosition = Vector3(0f, 0f, 1f),
            minPointCount = 2,
        )

        result.pose shouldBe null
        result.depthMeters shouldBe null
        result.details shouldBe "Need at least 2 points, but got 1"
    }

    @Test
    fun `plane fit succeeds for realistic non-collinear points`() {
        val points = listOf(
            Vector3(0f, 0f, 1f),
            Vector3(1f, 0f, 3f),
            Vector3(0f, 1f, 4f),
            Vector3(1f, 1f, 6f),
        )
        val cameraPosition = Vector3(0f, 0f, 10f)

        val result = fitPlanePoseFromPoints(
            worldPoints = points,
            cameraPosition = cameraPosition,
            minPointCount = 3,
        )

        result.details shouldBe "ok"
        result.depthMeters shouldNotBe null
        result.pose shouldNotBe null

        val pose = result.pose!!
        val toCamera = (cameraPosition - pose.center).normalized()
        (pose.normal.dot(toCamera) >= 0f) shouldBe true
        pose.normal.length shouldBe (1f plusOrMinus 0.0001f)
    }

    @Test
    fun `plane fit returns failure details when regression cannot be solved`() {
        val repeatedPoint = Vector3(2f, 2f, 2f)
        val result = fitPlanePoseFromPoints(
            worldPoints = listOf(repeatedPoint, repeatedPoint, repeatedPoint),
            cameraPosition = Vector3(0f, 0f, 10f),
            minPointCount = 3,
        )

        result.pose shouldBe null
        result.depthMeters shouldBe null
        result.details.isNotBlank() shouldBe true
        result.details shouldNotBe "ok"
    }
}
