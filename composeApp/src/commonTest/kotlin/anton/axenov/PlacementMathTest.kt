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
    fun `image-point sampling handles invalid sizes and clamped randomized bounds`() {
        val invalid = sampleImagePointsInScreenBoundingBox(
            screenBoundingBox = ScreenBoundingBox(0, 0, 10, 10),
            imageWidth = 0,
            imageHeight = 20,
            count = 4,
            random = Random(1),
        )
        invalid shouldBe emptyList()

        val points = sampleImagePointsInScreenBoundingBox(
            screenBoundingBox = ScreenBoundingBox(left = -10, top = -5, right = 8, bottom = 6),
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
        val points = sampleImagePointsInScreenBoundingBox(
            screenBoundingBox = ScreenBoundingBox(left = 3, top = 3, right = 3, bottom = 3),
            imageWidth = 10,
            imageHeight = 10,
            count = 0,
            random = Random(0),
        )

        points shouldBe listOf(ImagePoint(3, 3))
    }

    @Test
    fun `polygon sampling returns only points inside segmentation polygon`() {
        val polygon = listOf(
            ImagePoint(0, 0),
            ImagePoint(10, 0),
            ImagePoint(0, 10),
        )

        val points = sampleImagePointsInScreenPolygon(
            screenPolygon = polygon,
            imageWidth = 20,
            imageHeight = 20,
            count = 20,
            random = Random(42),
        )

        points.size shouldBe 20
        points.all { point -> isImagePointInsidePolygon(point, polygon) } shouldBe true
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
