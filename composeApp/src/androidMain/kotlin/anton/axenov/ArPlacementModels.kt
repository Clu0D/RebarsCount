package anton.axenov

import android.graphics.Bitmap
import com.google.ar.core.Pose
import io.github.sceneview.ar.node.AnchorNode

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
    val cameraPose: Pose,
    val depthSnapshot: DepthSnapshot?,
)

/**
 * Depth map sampled from one frame.
 *
 * Values are raw ARCore `DEPTH16` values.
 *
 * @param width depth image width.
 * @param height depth image height.
 * @param values depth values in row-major order.
 */
data class DepthSnapshot(
    val width: Int,
    val height: Int,
    val values: ShortArray,
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
 * Result of world placement attempt for one detected bounding box.
 *
 * @param anchorNode placed anchor node or null when placement failed.
 * @param details detailed diagnostic information.
 */
data class BoundingBoxPlacementResult(
    val anchorNode: AnchorNode?,
    val pointNodes: List<AnchorNode> = emptyList(),
    val details: String,
) {
    val placedNodes: List<AnchorNode>
        get() = listOfNotNull(anchorNode) + pointNodes
}
