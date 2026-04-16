package anton.axenov

import io.kotest.matchers.shouldBe
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.junit.jupiter.api.Test

/**
 * Tests for [TriangulationMath].
 */
class TriangulationMathTest {

    /**
     * Verifies points are filtered by distance to the computed epipolar line.
     */
    @Test
    fun `candidatesNearEpipolarLine should keep only points near the computed line`() {
        val triangulator = TriangulationMath()
        val fundamentalMatrix = Mat.zeros(3, 3, CvType.CV_64F)
        fundamentalMatrix.put(1, 2, 1.0)
        fundamentalMatrix.put(2, 2, -20.0)

        val candidates = triangulator.candidatesNearEpipolarLine(
            p1 = ImagePoint(5, 7),
            F = fundamentalMatrix,
            points2 = listOf(
                ImagePoint(100, 20),
                ImagePoint(150, 21),
                ImagePoint(120, 24),
            ),
            epsilonPx = 1.0,
        )

        candidates shouldBe setOf(
            0,
            1,
        )
    }

    /**
     * Verifies correspondence candidates are derived from snapshot geometry.
     */
    @Test
    fun `correspondenceCandidates should use snapshot geometry to build horizontal epipolar lines`() {
        val triangulator = TriangulationMath()
        val firstImagePoint = ImagePoint(320, 240)
        val result = triangulator.correspondenceCandidates(
            firstSnapshot = createSnapshot(
                translation = Vector3F(0f, 0f, 0f),
                rotationQuaternion = Quaternion.IDENTITY,
            ),
            secondSnapshot = createSnapshot(
                translation = Vector3F(1f, 0f, 0f),
                rotationQuaternion = Quaternion.IDENTITY,
            ),
            firstImagePoints = listOf(firstImagePoint),
            secondImagePoints = listOf(
                ImagePoint(120, 240),
                ImagePoint(520, 240),
                ImagePoint(250, 245),
            ),
            epsilonPx = 0.5,
        )

        result shouldBe listOf(
            setOf(0, 1),
        )
    }
}

/**
 * Creates one synthetic frame snapshot for triangulation tests.
 *
 * @param translation camera world translation.
 * @param rotationQuaternion camera world rotation.
 * @return snapshot with stable intrinsics.
 */
private fun createSnapshot(
    translation: Vector3F,
    rotationQuaternion: Quaternion,
): DetectionFrameSnapshotDto {
    return DetectionFrameSnapshotDto(
        screenshotPngBytes = byteArrayOf(),
        frameTimestamp = 1L,
        imageWidth = 640,
        imageHeight = 480,
        focalLengthX = 800f,
        focalLengthY = 800f,
        principalPointX = 320f,
        principalPointY = 240f,
        cameraPose = CameraPoseDto(
            translation = translation,
            rotationQuaternion = rotationQuaternion,
        ),
        depthSnapshot = null,
    )
}
