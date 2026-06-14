package anton.axenov

import kotlin.test.Test
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.shouldBe
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F

/**
 * Tests for [assignWorldPointsToZones].
 */
class WorldPointZoneSeparationTest {
    @Test
    fun `points deeply inside zones should become fixed anchors`() {
        val leftZone = separationZone(id = 1L, center = Vector3F(0f, 0f, 0f))
        val rightZone = separationZone(id = 2L, center = Vector3F(10f, 0f, 0f))
        val assignments = assignWorldPointsToZones(
            worldPoints = listOf(
                separationWorldPoint(x = 1f),
                separationWorldPoint(x = 9f),
            ),
            zones = listOf(leftZone, rightZone),
        )

        assignments shouldHaveSize 2
        assignments[0].zoneId shouldBe 1L
        assignments[0].isAnchor shouldBe true
        assignments[1].zoneId shouldBe 2L
        assignments[1].isAnchor shouldBe true
    }

    @Test
    fun `unlabelled boundary points should inherit labels from local neighbours`() {
        val leftZone = separationZone(id = 1L, center = Vector3F(0f, 0f, 0f))
        val rightZone = separationZone(id = 2L, center = Vector3F(10f, 0f, 0f))
        val assignments = assignWorldPointsToZones(
            worldPoints = listOf(
                separationWorldPoint(x = 1f),
                separationWorldPoint(x = 2.5f),
                separationWorldPoint(x = 4f),
                separationWorldPoint(x = 6f),
                separationWorldPoint(x = 7.5f),
                separationWorldPoint(x = 9f),
            ),
            zones = listOf(leftZone, rightZone),
            neighbourCount = 2,
        )

        assignments.map { assigned -> assigned.zoneId } shouldBe listOf(1L, 1L, 1L, 2L, 2L, 2L)
        assignments[2].isAnchor shouldBe false
        assignments[3].isAnchor shouldBe false
    }

    @Test
    fun `points without propagated neighbours should fall back to the nearest center`() {
        val leftZone = separationZone(id = 1L, center = Vector3F(0f, 0f, 0f))
        val rightZone = separationZone(id = 2L, center = Vector3F(10f, 0f, 0f))
        val assignments = assignWorldPointsToZones(
            worldPoints = listOf(separationWorldPoint(x = 4.9f, confidence = 0.9f)),
            zones = listOf(leftZone, rightZone),
        )

        assignments.single().zoneId shouldBe 1L
        assignments.single().isAnchor shouldBe false
    }
}

/**
 * Creates one synthetic zone for point-to-zone separation tests.
 *
 * @param id stable zone identifier.
 * @param center requested zone center.
 * @return test zone.
 */
private fun separationZone(
    id: Long,
    center: Vector3F,
): Zone {
    return Zone(
        id = id,
        sampledPoints = listOf(center),
        planePose = PlanePose(
            center = center,
            rotation = Quaternion.IDENTITY,
            normal = Vector3F(0f, 0f, 1f),
        ),
    )
}

/**
 * Creates one synthetic reconstructed world point.
 *
 * @param x X coordinate of the point.
 * @param confidence reconstruction confidence.
 * @return test world point.
 */
private fun separationWorldPoint(
    x: Float,
    confidence: Float = 1f,
): WorldPoint {
    return WorldPoint(
        position = Vector3F(x, 0f, 0f),
        parentPoints = emptySet(),
        confidence = confidence,
    )
}
