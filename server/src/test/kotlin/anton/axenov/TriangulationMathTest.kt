package anton.axenov

import io.kotest.matchers.doubles.plusOrMinus
import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F
import kotlin.test.Test

/**
 * Tests for [TriangulationMath].
 */
class TriangulationMathTest {

    /**
     * Verifies distorted observations still produce one correspondence when their world rays nearly intersect.
     */
    @Test
    fun `correspondenceCandidates should match distorted observations by ray distance`() {
        val triangulator = TriangulationMath()
        val distortion = listOf(0.14f, -0.03f, 0.002f, -0.001f, 0.008f)
        val firstSnapshot = createSnapshot(
            translation = Vector3F(-0.45f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
            distortionCoefficients = distortion,
        )
        val secondSnapshot = createSnapshot(
            translation = Vector3F(0.45f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
            distortionCoefficients = distortion,
        )
        val targetPoint = Vector3F(0f, 0.08f, -3.2f)

        val candidates = triangulator.correspondenceCandidates(
            firstSnapshot = firstSnapshot,
            secondSnapshot = secondSnapshot,
            firstImagePoints = listOf(projectPoint(firstSnapshot, targetPoint)),
            secondImagePoints = listOf(
                projectPoint(secondSnapshot, targetPoint),
                ImagePoint(50, 50),
                ImagePoint(600, 430),
            ),
            epsilonMeters = 0.05,
        )

        candidates.map { candidateList -> candidateList.map { it.first }.toSet() } shouldBe listOf(setOf(0))
        val midpoint = candidates.single().single().second
        (midpoint.distanceConfidence > 0.9f) shouldBe true
        (midpoint.angleConfidence > 0f) shouldBe true
        (midpoint.angleConfidence < 1f) shouldBe true
    }

    /**
     * Verifies near-parallel rays get lower angle confidence than wider-baseline rays.
     */
    @Test
    fun `correspondenceCandidates should reduce angle confidence for near parallel rays`() {
        val triangulator = TriangulationMath()
        val targetPoint = Vector3F(0f, 0f, -5f)
        val wideBaselineFirst = createSnapshot(
            translation = Vector3F(0f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
        )
        val wideBaselineSecond = createSnapshot(
            translation = Vector3F(1f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
        )
        val narrowBaselineSecond = createSnapshot(
            translation = Vector3F(0.2f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
        )

        val wideBaselineCandidate = triangulator.correspondenceCandidates(
            firstSnapshot = wideBaselineFirst,
            secondSnapshot = wideBaselineSecond,
            firstImagePoints = listOf(projectPoint(wideBaselineFirst, targetPoint)),
            secondImagePoints = listOf(projectPoint(wideBaselineSecond, targetPoint)),
            epsilonMeters = 0.05,
        ).single().single().second
        val narrowBaselineCandidate = triangulator.correspondenceCandidates(
            firstSnapshot = wideBaselineFirst,
            secondSnapshot = narrowBaselineSecond,
            firstImagePoints = listOf(projectPoint(wideBaselineFirst, targetPoint)),
            secondImagePoints = listOf(projectPoint(narrowBaselineSecond, targetPoint)),
            epsilonMeters = 0.05,
        ).single().single().second

        (narrowBaselineCandidate.angleConfidence < wideBaselineCandidate.angleConfidence) shouldBe true
    }

    /**
     * Verifies the hard rejection threshold can preserve a neutral pair without accepting it as support.
     */
    @Test
    fun `correspondenceCandidates should retain pairs up to explicit forbidden distance`() {
        val triangulator = TriangulationMath()
        val firstSnapshot = createSnapshot(
            translation = Vector3F(0f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
        )
        val secondSnapshot = createSnapshot(
            translation = Vector3F(1f, 0f, 0f),
            rotationQuaternion = Quaternion.IDENTITY,
        )
        val firstPoint = projectPoint(firstSnapshot, Vector3F(0f, 0f, -5f))
        val shiftedSecondPoint = projectPoint(secondSnapshot, Vector3F(0f, 0.1f, -5f))

        val supportCandidates = triangulator.correspondenceCandidates(
            firstSnapshot = firstSnapshot,
            secondSnapshot = secondSnapshot,
            firstImagePoints = listOf(firstPoint),
            secondImagePoints = listOf(shiftedSecondPoint),
            epsilonMeters = 0.05,
        )
        val nonForbiddenCandidates = triangulator.correspondenceCandidates(
            firstSnapshot = firstSnapshot,
            secondSnapshot = secondSnapshot,
            firstImagePoints = listOf(firstPoint),
            secondImagePoints = listOf(shiftedSecondPoint),
            epsilonMeters = 0.05,
            maxDistanceMeters = 0.2,
        )

        supportCandidates.single().isEmpty() shouldBe true
        nonForbiddenCandidates.single().single().second.distance shouldBe (0.1 plusOrMinus 0.01)
    }
}

/**
 * Creates one synthetic frame snapshot for triangulation tests.
 *
 * @param translation camera world translation.
 * @param rotationQuaternion camera world rotation.
 * @param imageWidth image width in pixels.
 * @param imageHeight image height in pixels.
 * @param focalLengthX focal length in pixels along X.
 * @param focalLengthY focal length in pixels along Y.
 * @param principalPointX principal point X in pixels.
 * @param principalPointY principal point Y in pixels.
 * @param distortionCoefficients lens distortion coefficients in OpenCV order.
 * @return snapshot with stable intrinsics.
 */
private fun createSnapshot(
    translation: Vector3F,
    rotationQuaternion: Quaternion,
    imageWidth: Int = 640,
    imageHeight: Int = 480,
    focalLengthX: Float = 800f,
    focalLengthY: Float = 800f,
    principalPointX: Float = 320f,
    principalPointY: Float = 240f,
    distortionCoefficients: List<Float> = emptyList(),
): DetectionFrameSnapshotDto {
    return DetectionFrameSnapshotDto(
        screenshotPngBytes = byteArrayOf(),
        frameTimestamp = 1L,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        focalLengthX = focalLengthX,
        focalLengthY = focalLengthY,
        principalPointX = principalPointX,
        principalPointY = principalPointY,
        distortionCoefficients = distortionCoefficients,
        cameraPose = CameraPoseDto(
            translation = translation,
            rotationQuaternion = rotationQuaternion,
        ),
        depthSnapshot = null,
    )
}

/**
 * Builds camera orientation that maps local camera forward `-Z` toward world origin.
 *
 * @param cameraPosition world-space camera position exactly 1 meter from origin.
 * @return camera-to-world rotation quaternion looking at origin.
 */
private fun lookAtOriginQuaternion(cameraPosition: Vector3F): Quaternion {
    return Quaternion.fromVectors(
        Vector3F(0f, 0f, -1f),
        (-cameraPosition).normalized(),
    ).normalized()
}

/**
 * Projects one world-space point into one distorted image.
 *
 * @param snapshot camera snapshot with intrinsics and distortion.
 * @param worldPoint world-space point to project.
 * @return distorted image point.
 */
private fun projectPoint(
    snapshot: DetectionFrameSnapshotDto,
    worldPoint: Vector3F,
): ImagePoint {
    val worldRotation = snapshot.cameraPose.rotationQuaternion.normalized()
    val cameraPoint = worldRotation
        .inverted()
        .transform(worldPoint - snapshot.cameraPose.translation)
    val normalizedX = cameraPoint.x / cameraPoint.z
    val normalizedY = cameraPoint.y / cameraPoint.z
    val distorted = distortNormalizedPoint(
        x = normalizedX.toDouble(),
        y = normalizedY.toDouble(),
        distortionCoefficients = snapshot.distortionCoefficients,
    )
    return ImagePoint(
        x = (
                snapshot.focalLengthX * distorted.first +
                        snapshot.principalPointX
                ).toInt(),
        y = (
                snapshot.focalLengthY * distorted.second +
                        snapshot.principalPointY
                ).toInt(),
    )
}

/**
 * Applies OpenCV radial-tangential distortion to one normalized camera point.
 *
 * @param x normalized camera X.
 * @param y normalized camera Y.
 * @param distortionCoefficients lens distortion coefficients in OpenCV order.
 * @return distorted normalized point.
 */
private fun distortNormalizedPoint(
    x: Double,
    y: Double,
    distortionCoefficients: List<Float>,
): Pair<Double, Double> {
    val k1 = distortionCoefficients.getOrElse(0) { 0f }.toDouble()
    val k2 = distortionCoefficients.getOrElse(1) { 0f }.toDouble()
    val p1 = distortionCoefficients.getOrElse(2) { 0f }.toDouble()
    val p2 = distortionCoefficients.getOrElse(3) { 0f }.toDouble()
    val k3 = distortionCoefficients.getOrElse(4) { 0f }.toDouble()
    val r2 = x * x + y * y
    val radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    val distortedX = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    val distortedY = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return distortedX to distortedY
}
