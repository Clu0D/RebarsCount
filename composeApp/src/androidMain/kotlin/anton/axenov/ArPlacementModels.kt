package anton.axenov

import android.graphics.Bitmap
import com.google.ar.core.Pose

/**
 * Snapshot of one frame used for asynchronous detection and world placement.
 *
 * @param screenshot screenshot bitmap captured from this frame.
 * @param frameTimestamp frame timestamp in nanoseconds.
 * @param imageWidth camera image width in pixels.
 * @param imageHeight camera image height in pixels.
 * @param focalLengthX camera focal length X in pixels.
 * @param focalLengthY camera focal length Y in pixels.
 * @param principalPointX camera principal point X in pixels.
 * @param principalPointY camera principal point Y in pixels.
 * @param distortionCoefficients lens distortion coefficients.
 * @param cameraPose camera pose for this exact frame.
 * @param depthSnapshot depth values sampled from this exact frame or null when depth is unavailable.
 */
data class DetectionFrameSnapshot(
    val screenshot: Bitmap,
    val frameTimestamp: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val focalLengthX: Float,
    val focalLengthY: Float,
    val principalPointX: Float,
    val principalPointY: Float,
    val distortionCoefficients: List<Float>,
    val cameraPose: Pose,
    val depthSnapshot: DepthSnapshot?,
)

/**
 * Result of snapshot capture attempt.
 *
 * @param snapshot captured frame snapshot or null when capture failed.
 * @param details capture diagnostic details.
 */
data class DetectionFrameSnapshotCaptureResult(
    val snapshot: DetectionFrameSnapshot?,
    val details: String,
)

/**
 * Result of world placement attempt for one detected zone.
 *
 * @param zone placed zone data or null when placement failed.
 * @param details detailed diagnostic information.
 */
data class ZonePlacementResult(
    val zone: Zone?,
    val details: String,
)
