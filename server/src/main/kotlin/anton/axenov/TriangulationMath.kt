package anton.axenov

import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F
import kotlin.math.abs
import kotlin.math.sqrt
import nu.pattern.OpenCV
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import korlibs.math.geom.Vector3F as Vector3

/**
 * Finds epipolar-consistent point candidates between two frame snapshots.
 */
class TriangulationMath {

    companion object {
        init {
            OpenCV.loadLocally()
        }
    }

    /**
     * Finds all points from [points2] that can correspond to [p1]
     * according to the epipolar geometry between 2 frames.
     *
     * @param p1 point from the first image.
     * @param F fundamental matrix between frames.
     * @param points2 candidate points from the second image.
     * @param epsilonPx maximal distance in pixels from the epipolar line.
     * @return set of all points index from the second image that satisfy the epipolar constraint.
     */
    fun candidatesNearEpipolarLine(
        p1: ImagePoint,
        F: Mat,
        points2: List<ImagePoint>,
        epsilonPx: Double,
    ): Set<Int> {
        val pts1 = MatOfPoint2f(Point(p1.x.toDouble(), p1.y.toDouble()))
        val lines2 = Mat()
        Calib3d.computeCorrespondEpilines(pts1, 1, F, lines2)

        val line = lines2.get(0, 0)
        val a = line[0]
        val b = line[1]
        val c = line[2]
        val denom = sqrt(a * a + b * b)
        if (denom == 0.0) {
            return emptySet()
        }

        return points2.indices.filter { index ->
            val point = points2[index]
            abs(a * point.x.toDouble() + b * point.y.toDouble() + c) / denom <= epsilonPx
        }.toSet()
    }

    /**
     * Finds epipolar-consistent correspondence candidates for every point from [firstImagePoints].
     *
     * @param firstSnapshot first frame snapshot with camera intrinsics and pose.
     * @param secondSnapshot second frame snapshot with camera intrinsics and pose.
     * @param firstImagePoints points from the first image.
     * @param secondImagePoints candidate points from the second image.
     * @param epsilonPx maximal distance in pixels from the epipolar line.
     * @return map from each first-image point to all second-image points that can correspond to it.
     */
    fun correspondenceCandidates(
        firstSnapshot: DetectionFrameSnapshotDto,
        secondSnapshot: DetectionFrameSnapshotDto,
        firstImagePoints: List<ImagePoint>,
        secondImagePoints: List<ImagePoint>,
        epsilonPx: Double,
    ): List<Set<Int>> {
        val fundamentalMatrix = buildFundamentalMatrix(firstSnapshot, secondSnapshot)
        return firstImagePoints.map { firstImagePoint ->
            candidatesNearEpipolarLine(
                p1 = firstImagePoint,
                F = fundamentalMatrix,
                points2 = secondImagePoints,
                epsilonPx = epsilonPx,
            )
        }
    }

    /**
     * OpenCV triangulation reconstructs one world-space point from a pair of image points
     * from different frames as if they were corresponding.
     *
     * @param firstSnapshot first frame snapshot with camera intrinsics and pose.
     * @param secondSnapshot second frame snapshot with camera intrinsics and pose.
     * @param firstImagePoint point from the first image.
     * @param secondImagePoint corresponding point from the second image.
     * @return reconstructed world-space point or null if bad triangulation.
     */
    fun triangulateWorldPoint(
        firstSnapshot: DetectionFrameSnapshotDto,
        secondSnapshot: DetectionFrameSnapshotDto,
        firstImagePoint: ImagePoint,
        secondImagePoint: ImagePoint,
    ): Vector3? {
        val firstProjection = projectionMatrix(firstSnapshot)
        val secondProjection = projectionMatrix(secondSnapshot)
        val firstPoints = Mat(2, 1, CvType.CV_64F)
        firstPoints.put(0, 0, firstImagePoint.x.toDouble())
        firstPoints.put(1, 0, firstImagePoint.y.toDouble())
        val secondPoints = Mat(2, 1, CvType.CV_64F)
        secondPoints.put(0, 0, secondImagePoint.x.toDouble())
        secondPoints.put(1, 0, secondImagePoint.y.toDouble())

        val homogeneousPoint = Mat()
        Calib3d.triangulatePoints(
            firstProjection,
            secondProjection,
            firstPoints,
            secondPoints,
            homogeneousPoint,
        )
        if (homogeneousPoint.cols() == 0 || homogeneousPoint.rows() < 4) {
            return null
        }
        val w = homogeneousPoint.get(3, 0)[0]
        if (!w.isFinite() || abs(w) <= TRIANGULATION_W_EPSILON) {
            return null
        }
        val x = homogeneousPoint.get(0, 0)[0] / w
        val y = homogeneousPoint.get(1, 0)[0] / w
        val z = homogeneousPoint.get(2, 0)[0] / w
        if (!x.isFinite() || !y.isFinite() || !z.isFinite()) {
            return null
        }
        return Vector3(x.toFloat(), y.toFloat(), z.toFloat())
    }

    /**
     * Builds the fundamental matrix from 2 camera snapshots.
     *
     * @param frame1 first camera frame.
     * @param frame2 second camera frame.
     * @return fundamental matrix that maps frame-1 points to frame-2 epipolar lines.
     */
    private fun buildFundamentalMatrix(
        frame1: DetectionFrameSnapshotDto,
        frame2: DetectionFrameSnapshotDto,
    ): Mat {
        val q1 = frame1.cameraPose.rotationQuaternion.normalized()
        val q2 = frame2.cameraPose.rotationQuaternion.normalized()

        val q21 = q2.inverted() * q1
        val t21 = q2.inverted().transform(frame1.cameraPose.translation - frame2.cameraPose.translation)

        val R = rotationMatFromQuaternion(q21)
        val tx = skew(t21.x.toDouble(), t21.y.toDouble(), t21.z.toDouble())

        val E = Mat()
        Core.gemm(tx, R, 1.0, Mat(), 0.0, E)

        val K1 = cameraMatrix(frame1)
        val K2 = cameraMatrix(frame2)

        val tmp = Mat()
        val F = Mat()
        Core.gemm(K2.inv().t(), E, 1.0, Mat(), 0.0, tmp)
        Core.gemm(tmp, K1.inv(), 1.0, Mat(), 0.0, F)
        return F
    }

    /**
     * Builds one camera projection matrix.
     *
     * @param frame frame snapshot with intrinsics and camera pose.
     * @return projection matrix that maps world points to image points.
     */
    private fun projectionMatrix(frame: DetectionFrameSnapshotDto): Mat {
        val worldRotation = rotationMatFromQuaternion(frame.cameraPose.rotationQuaternion.normalized())
        val cameraRotation = worldRotation.t()
        val cameraCenter = frame.cameraPose.translation
        val translation = Mat(3, 1, CvType.CV_64F)
        translation.put(
            0,
            0,
            cameraCenter.x.toDouble(),
            cameraCenter.y.toDouble(),
            cameraCenter.z.toDouble(),
        )
        val cameraTranslation = Mat()
        Core.gemm(cameraRotation, translation, -1.0, Mat(), 0.0, cameraTranslation)

        val extrinsic = Mat(3, 4, CvType.CV_64F)
        for (row in 0 until 3) {
            for (col in 0 until 3) {
                extrinsic.put(row, col, cameraRotation.get(row, col)[0])
            }
            extrinsic.put(row, 3, cameraTranslation.get(row, 0)[0])
        }

        val projection = Mat()
        Core.gemm(cameraMatrix(frame), extrinsic, 1.0, Mat(), 0.0, projection)
        return projection
    }

    private fun cameraMatrix(frame: DetectionFrameSnapshotDto): Mat {
        val cameraMatrix = Mat.eye(3, 3, CvType.CV_64F)
        cameraMatrix.put(0, 0, frame.focalLengthX.toDouble())
        cameraMatrix.put(1, 1, frame.focalLengthY.toDouble())
        cameraMatrix.put(0, 2, frame.principalPointX.toDouble())
        cameraMatrix.put(1, 2, frame.principalPointY.toDouble())
        return cameraMatrix
    }

    private fun rotationMatFromQuaternion(q: Quaternion): Mat {
        q.toEuler()
        val ex = q.transform(Vector3F(1f, 0f, 0f))
        val ey = q.transform(Vector3F(0f, 1f, 0f))
        val ez = q.transform(Vector3F(0f, 0f, 1f))

        // columns = rotated basis vectors
        val R = Mat(3, 3, CvType.CV_64F)
        R.put(
            0, 0,
            ex.x.toDouble(), ey.x.toDouble(), ez.x.toDouble(),
            ex.y.toDouble(), ey.y.toDouble(), ez.y.toDouble(),
            ex.z.toDouble(), ey.z.toDouble(), ez.z.toDouble()
        )
        return R
    }

    private fun skew(x: Double, y: Double, z: Double): Mat {
        val m = Mat.zeros(3, 3, CvType.CV_64F)
        m.put(
            0, 0,
            0.0, -z, y,
            z, 0.0, -x,
            -y, x, 0.0
        )
        return m
    }
}

private const val TRIANGULATION_W_EPSILON = 1e-9
