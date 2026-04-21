package anton.axenov

import kotlin.math.abs
import nu.pattern.OpenCV
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
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
     * Finds epipolar-consistent correspondence candidates for every point from [firstImagePoints].
     *
     * @param firstSnapshot first frame snapshot with camera intrinsics and pose.
     * @param secondSnapshot second frame snapshot with camera intrinsics and pose.
     * @param firstImagePoints points from the first image.
     * @param secondImagePoints candidate points from the second image.
     * @param epsilonMeters maximal distance in meters between the two viewing rays.
     * @return map from each first-image point to all second-image points that can correspond to it.
     */
    fun correspondenceCandidates(
        firstSnapshot: DetectionFrameSnapshotDto,
        secondSnapshot: DetectionFrameSnapshotDto,
        firstImagePoints: List<ImagePoint>,
        secondImagePoints: List<ImagePoint>,
        epsilonMeters: Double,
    ): List<List<Pair<Int, RaysMidPoint>>> {
        val firstRays = firstImagePoints.map { imagePoint ->
            worldRay(firstSnapshot, imagePoint)
        }
        val secondRays = secondImagePoints.map { imagePoint ->
            worldRay(secondSnapshot, imagePoint)
        }
        return firstRays.map { firstRay ->
            secondRays.mapIndexedNotNull { index, secondRay ->
                val raysMidPoint = raysMidPoint(
                    firstRay = firstRay,
                    secondRay = secondRay,
                    distanceEpsilonMeters = epsilonMeters,
                ) ?: return@mapIndexedNotNull null
                Pair(index, raysMidPoint).takeIf { raysMidPoint.distance <= epsilonMeters }
            }
        }
    }

    /**
     * Builds one world-space viewing ray for [imagePoint].
     *
     * @param frame frame snapshot with intrinsics, distortion and pose.
     * @param imagePoint distorted image-space point.
     * @return world-space viewing ray leaving the camera center.
     */
    private fun worldRay(frame: DetectionFrameSnapshotDto, imagePoint: ImagePoint): CameraRay {
        val undistortedPoint = undistortImagePoint(frame, imagePoint)
        val cameraDirection = Vector3(
            -undistortedPoint.x.toFloat(),
            -undistortedPoint.y.toFloat(),
            -1f,
        ).normalized()
        val worldDirection = frame.cameraPose.rotationQuaternion
            .normalized()
            .transform(cameraDirection)
            .normalized()
        return CameraRay(
            origin = frame.cameraPose.translation,
            direction = worldDirection,
        )
    }

    /**
     * Removes lens distortion from one pixel observation and returns normalized camera coordinates.
     *
     * @param frame frame snapshot with camera intrinsics and distortion.
     * @param imagePoint distorted image-space point.
     * @return undistorted normalized camera coordinates.
     */
    private fun undistortImagePoint(frame: DetectionFrameSnapshotDto, imagePoint: ImagePoint): Point {
        val distortedPoints = MatOfPoint2f(
            Point(imagePoint.x.toDouble(), imagePoint.y.toDouble()),
        )
        val undistortedPoints = MatOfPoint2f()
        Calib3d.undistortPoints(
            distortedPoints,
            undistortedPoints,
            cameraMatrix(frame),
            MatOfDouble(*frame.distortionCoefficients.map { it.toDouble() }.toDoubleArray()),
        )
        return undistortedPoints.toArray().single()
    }

    /**
     * Finds the minimal distance between two forward viewing rays and point between them.
     *
     * @param firstRay first world-space viewing ray.
     * @param secondRay second world-space viewing ray.
     * @param distanceEpsilonMeters max accepted physical distance between rays.
     * @return distance between 2 rays and point between them.
     */
    private fun raysMidPoint(
        firstRay: CameraRay,
        secondRay: CameraRay,
        distanceEpsilonMeters: Double,
    ): RaysMidPoint? {
        val delta = firstRay.origin - secondRay.origin
        val a = firstRay.direction.dot(firstRay.direction)
        val b = firstRay.direction.dot(secondRay.direction)
        val c = secondRay.direction.dot(secondRay.direction)
        val d = firstRay.direction.dot(delta)
        val e = secondRay.direction.dot(delta)
        val denominator = a * c - b * b
        if (abs(denominator) <= RAY_PARALLEL_EPSILON)
            return null

        // Solve the closest-point system on infinite lines
        // Real world distances, as the direction is normalized
        val firstDistance = ((b * e) - (c * d)) / denominator
        val secondDistance = ((a * e) - (b * d)) / denominator
        // reject points behind cameras or too far away
        if (firstDistance <= MIN_RAY_DEPTH_METERS || secondDistance <= MIN_RAY_DEPTH_METERS ||
            firstDistance >= MAX_TRIANGULATION_DISTANCE_METERS || secondDistance >= MAX_TRIANGULATION_DISTANCE_METERS
        )
            return null

        val firstPoint = firstRay.origin + firstRay.direction * firstDistance
        val secondPoint = secondRay.origin + secondRay.direction * secondDistance

        val distance = (firstPoint - secondPoint).length.toDouble()
        val midPoint = (firstPoint + secondPoint) * 0.5f

        return RaysMidPoint(
            distance = distance,
            midPoint = midPoint,
            distanceConfidence = distanceConfidence(distance, distanceEpsilonMeters),
            angleConfidence = angleConfidence(firstRay.direction, secondRay.direction),
        )
    }

    /**
     * Midpoint between 2 rays with confidence generated while triangulation
     */
    data class RaysMidPoint(
        val distance: Double,
        val midPoint: Vector3,
        val distanceConfidence: Float,
        val angleConfidence: Float,
    )

    /**
     * World-space ray from camera.
     */
    private data class CameraRay(
        val origin: Vector3,
        val direction: Vector3,
    )

    private fun cameraMatrix(frame: DetectionFrameSnapshotDto): Mat {
        val cameraMatrix = Mat.eye(3, 3, CvType.CV_64F)
        cameraMatrix.put(0, 0, frame.focalLengthX.toDouble())
        cameraMatrix.put(1, 1, frame.focalLengthY.toDouble())
        cameraMatrix.put(0, 2, frame.principalPointX.toDouble())
        cameraMatrix.put(1, 2, frame.principalPointY.toDouble())
        return cameraMatrix
    }

    private fun distanceConfidence(
        distance: Double,
        distanceEpsilonMeters: Double,
    ): Float {
        return ((1.0 - (distance / distanceEpsilonMeters)) * (1.0 + DISTANCE_CONFIDENCE_COEFFICIENT))
            .toFloat()
            .coerceIn(0f, 1f)
    }

    private fun angleConfidence(
        firstDirection: Vector3,
        secondDirection: Vector3,
    ): Float {
        return (firstDirection.cross(secondDirection).length * ANGLE_CONFIDENCE_COEFFICIENT)
            .coerceIn(0f, 1f)
    }
}

private const val RAY_PARALLEL_EPSILON = 1e-6
private const val MIN_RAY_DEPTH_METERS = 0.1f
private const val MAX_TRIANGULATION_DISTANCE_METERS = 10f
private const val DISTANCE_CONFIDENCE_COEFFICIENT = 1.5f
private const val ANGLE_CONFIDENCE_COEFFICIENT = 2f
