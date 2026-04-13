package anton.axenov

import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F
import kotlinx.serialization.Serializable

/**
 * Health response returned by the segmentation server.
 *
 * @param ok true when server is healthy.
 * @param message user-visible server message.
 */
@Serializable
data class ServerHealthResponse(
    val ok: Boolean,
    val message: String,
)

/**
 * Status text returned for one zone.
 *
 * @param zone stable zone identifier.
 * @param text user-visible zone status text.
 */
@Serializable
data class ZoneStatus(
    val zone: Long,
    val text: String,
)

/**
 * Response returned after one snapshot upload.
 *
 * @param ok true when snapshot was stored.
 * @param zoneId uploaded zone identifier.
 * @param snapshotCount stored snapshot count for this zone.
 * @param message user-visible server message.
 */
@Serializable
data class SnapshotUploadResponse(
    val ok: Boolean,
    val zoneId: Long,
    val snapshotCount: Int,
    val message: String,
)

/**
 * Bounding box from python response.
 */
@Serializable
data class SegmentationBoundingBox(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
)

/**
 * Segmented object instance returned by Python service.
 */
@Serializable
data class SegmentationInstance(
    val id: Int,
    val bbox: SegmentationBoundingBox,
)

/**
 * Prediction payload returned by Python segmentation service.
 *
 * @param filename source filename echoed by Python service.
 * @param width image width in pixels.
 * @param height image height in pixels.
 * @param count number of detected instances.
 * @param instances detected object instances.
 */
@Serializable
data class SegmentationPrediction(
    val filename: String,
    val width: Int,
    val height: Int,
    val count: Int,
    val instances: List<SegmentationInstance>,
)

/**
 * Current processing state of one stored snapshot on the Ktor server.
 */
@Serializable
enum class SegmentationState {
    QUEUED,
    PROCESSING,
    COMPLETED,
    FAILED,
}

/**
 * Serializable camera pose.
 *
 * @param translation camera translation.
 * @param rotationQuaternion camera rotation quaternion.
 */
@Serializable
data class CameraPoseDto(
    @Serializable(with = Vector3Serializer::class)
    val translation: Vector3F,
    @Serializable(with = QuaternionSerializer::class)
    val rotationQuaternion: Quaternion,
)

/**
 * Serializable frame snapshot payload.
 *
 * @param screenshotPngBytes screenshot encoded as PNG bytes.
 * @param frameTimestamp frame timestamp in nanoseconds.
 * @param imageWidth camera image width.
 * @param imageHeight camera image height.
 * @param focalLengthX camera focal length X.
 * @param focalLengthY camera focal length Y.
 * @param principalPointX camera principal point X.
 * @param principalPointY camera principal point Y.
 * @param cameraPose camera pose.
 * @param depthSnapshot optional depth snapshot.
 */
@Serializable
data class DetectionFrameSnapshotDto(
    val screenshotPngBytes: ByteArray,
    val frameTimestamp: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val focalLengthX: Float,
    val focalLengthY: Float,
    val principalPointX: Float,
    val principalPointY: Float,
    val cameraPose: CameraPoseDto,
    val depthSnapshot: DepthSnapshot?,
)

/**
 * Upload payload for one stored zone snapshot.
 *
 * @param zone serialized zone payload.
 * @param frameSnapshot serialized frame payload.
 * @param captureAngle serialized capture-angle metrics.
 * @param screenCoverage serialized screen-coverage metrics.
 */
@Serializable
data class ZoneSnapshotUploadDto(
    val zone: Zone,
    val frameSnapshot: DetectionFrameSnapshotDto,
    val captureAngle: ZoneCaptureAngle,
    val screenCoverage: ZoneScreenCoverageMetrics,
)
