package anton.axenov

import kotlin.test.Test
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F

/**
 * Tests for resolved-component dissolution and ordinary-pipeline reassembly.
 */
class ComponentReassemblyTest {
    @Test
    fun `component confidence should reward support edge count without clamping`() {
        val resolver = WorldPointHypothesisResolver()
        val observations = reassemblyObservations(startIndex = 40)
        val component = reassemblyComponent(
            resolver = resolver,
            observations = observations,
            position = Vector3F(0f, 0f, 0f),
            confidence = 1f,
        )

        component.confidence shouldBe (2f plusOrMinus 0.0001f)
    }

    @Test
    fun `component confidence should combine average edge weight with size bonus`() {
        val resolver = WorldPointHypothesisResolver()
        val observations = reassemblyObservations(startIndex = 50)
        val component = resolver.HypothesisComponent(
            reassemblyEdge(observations[0], observations[1], Vector3F(0f, 0f, 0f), confidence = 0.2f),
        )
        component.addObservation(
            observation = observations[2],
            supportEdges = listOf(
                reassemblyEdge(observations[0], observations[2], Vector3F(0f, 0f, 0f), confidence = 0.4f),
                reassemblyEdge(observations[1], observations[2], Vector3F(0f, 0f, 0f), confidence = 0.6f),
            ),
        )
        component.addObservation(
            observation = observations[3],
            supportEdges = listOf(
                reassemblyEdge(observations[0], observations[3], Vector3F(0f, 0f, 0f), confidence = 0.8f),
            ),
        )

        component.recomputeCenterAndConfidence()

        component.confidence shouldBe (1f plusOrMinus 0.0001f)
    }

    @Test
    fun `minimum component should preserve required average support after size bonus`() {
        val resolver = WorldPointHypothesisResolver()
        val weakComponent = reassemblyComponent(
            resolver = resolver,
            observations = reassemblyObservations(startIndex = 60),
            position = Vector3F(0f, 0f, 0f),
            confidence = 0.7f,
        )
        val strongComponent = reassemblyComponent(
            resolver = resolver,
            observations = reassemblyObservations(startIndex = 70),
            position = Vector3F(0f, 0f, 0f),
            confidence = 0.8f,
        )

        weakComponent.isBad() shouldBe true
        strongComponent.isBad() shouldBe false
    }

    @Test
    fun `ray-selected component should receive observation before ordinary point processing`() {
        val resolver = WorldPointHypothesisResolver()
        val componentObservations = reassemblyObservations(startIndex = 80)
        val component = reassemblyComponent(
            resolver = resolver,
            observations = componentObservations,
            position = Vector3F(0f, 0f, 0f),
        )
        val newObservation = reassemblyObservations(startIndex = 90).first()
        val support = reassemblyEdge(
            first = componentObservations.first(),
            second = newObservation,
            position = Vector3F(0f, 0f, 0f),
        )
        resolver.resolvedComponents += component

        val resolved = resolver.resolve(
            newWorldPoints = listOf(support),
            candidateComponentsByObservation = mapOf(newObservation to listOf(component)),
        )

        resolved shouldHaveSize 1
        resolved.single().parentPoints shouldHaveSize 5
        (newObservation in resolved.single().parentPoints) shouldBe true
    }

    @Test
    fun `close components should dissolve and glue again through ordinary resolve pipeline`() {
        val resolver = WorldPointHypothesisResolver(
            componentDissolveDistanceMeters = 0.01f,
            maxFreeWorldPoints = 100,
        )
        val firstPoints = reassemblyObservations(startIndex = 0)
        val secondPoints = reassemblyObservations(startIndex = 10)
        val firstComponent = reassemblyComponent(
            resolver = resolver,
            observations = firstPoints,
            position = Vector3F(0f, 0f, 0f),
        )
        val secondComponent = reassemblyComponent(
            resolver = resolver,
            observations = secondPoints,
            position = Vector3F(0.005f, 0f, 0f),
        )
        resolver.resolvedComponents += firstComponent
        resolver.resolvedComponents += secondComponent

        val resolved = resolver.resolve(
            listOf(
                reassemblyEdge(
                    first = firstPoints.first(),
                    second = secondPoints.first(),
                    position = Vector3F(0.0025f, 0f, 0f),
                ),
            ),
        )

        resolved shouldHaveSize 1
        resolved.single().parentPoints shouldHaveSize 8
        resolver.resolvedComponents shouldHaveSize 1
    }

    @Test
    fun `audit should dissolve weakest excess components`() {
        val resolver = WorldPointHypothesisResolver(
            componentDissolveDistanceMeters = 0f,
            maxFreeWorldPoints = 100,
        )
        val strongComponent = reassemblyComponent(
            resolver = resolver,
            observations = reassemblyObservations(startIndex = 20),
            position = Vector3F(0f, 0f, 0f),
            confidence = 1f,
        )
        val weakComponent = reassemblyComponent(
            resolver = resolver,
            observations = reassemblyObservations(startIndex = 30),
            position = Vector3F(1f, 0f, 0f),
            confidence = 0.8f,
        )
        resolver.resolvedComponents += strongComponent
        resolver.resolvedComponents += weakComponent

        val released = resolver.auditAndDissolveComponents()

        resolver.resolvedComponents shouldHaveSize 1
        resolver.resolvedComponents.single() shouldBe strongComponent
        released.toSet() shouldBe weakComponent.curPoints()
    }
}

/**
 * Creates one valid synthetic component with four supporting edges.
 *
 * @param resolver resolver that owns the component.
 * @param observations four observations from different frames.
 * @param position shared edge position.
 * @param confidence edge confidence.
 * @return valid component.
 */
private fun reassemblyComponent(
    resolver: WorldPointHypothesisResolver,
    observations: List<ZoneTriangulationPoint>,
    position: Vector3F,
    confidence: Float = 1f,
): WorldPointHypothesisResolver.HypothesisComponent {
    val firstEdge = reassemblyEdge(observations[0], observations[1], position, confidence)
    val component = resolver.HypothesisComponent(firstEdge)
    component.addObservation(
        observation = observations[2],
        supportEdges = listOf(
            reassemblyEdge(observations[0], observations[2], position, confidence),
            reassemblyEdge(observations[1], observations[2], position, confidence),
        ),
    )
    component.addObservation(
        observation = observations[3],
        supportEdges = listOf(reassemblyEdge(observations[0], observations[3], position, confidence)),
    )
    component.recomputeCenterAndConfidence()
    return component
}

/**
 * Creates four observations from independent frames.
 *
 * @param startIndex first segmentation index.
 * @return four synthetic observations.
 */
private fun reassemblyObservations(startIndex: Int): List<ZoneTriangulationPoint> {
    return (startIndex until startIndex + 4).map { index ->
        ZoneTriangulationPoint(
            segmentation = reassemblySegmentation(index),
            imagePoint = ImagePoint(index, index),
            confidence = 1f,
        )
    }
}

/**
 * Creates one pairwise support edge.
 *
 * @param first first observation.
 * @param second second observation.
 * @param position edge position.
 * @param confidence edge confidence.
 * @return synthetic edge.
 */
private fun reassemblyEdge(
    first: ZoneTriangulationPoint,
    second: ZoneTriangulationPoint,
    position: Vector3F,
    confidence: Float = 1f,
): WorldPoint {
    return WorldPoint(
        position = position,
        parentPoints = setOf(first, second),
        confidence = confidence,
    )
}

/**
 * Creates one minimal segmentation for component-reassembly tests.
 *
 * @param index unique segmentation index.
 * @return synthetic segmentation.
 */
private fun reassemblySegmentation(index: Int): ZoneSegmentation {
    return ZoneSegmentation(
        segmentationIndex = index,
        zone = reassemblyZone(),
        frameSnapshot = DetectionFrameSnapshotDto(
            screenshotPngBytes = byteArrayOf(),
            frameTimestamp = index.toLong(),
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
        ),
        prediction = SegmentationPrediction(
            filename = "$index.png",
            width = 1,
            height = 1,
            count = 1,
            instances = emptyList(),
        ),
    )
}

/**
 * Creates one minimal zone for component-reassembly tests.
 *
 * @return synthetic zone.
 */
private fun reassemblyZone(): Zone {
    return Zone(
        id = 1L,
        sampledPoints = listOf(Vector3F(0f, 0f, 0f)),
        planePose = PlanePose(
            center = Vector3F(0f, 0f, 0f),
            rotation = Quaternion.IDENTITY,
            normal = Vector3F(0f, 1f, 0f),
        ),
    )
}
