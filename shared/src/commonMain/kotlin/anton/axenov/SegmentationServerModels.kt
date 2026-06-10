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
 * @param total total snapshots accepted for the zone.
 * @param queued snapshots waiting for processing.
 * @param processing snapshots currently being processed.
 * @param completed successfully processed snapshots.
 * @param failed snapshots that failed processing.
 */
@Serializable
data class ZoneStatus(
    val zone: Long,
    val text: String,
    val total: Int = 0,
    val queued: Int = 0,
    val processing: Int = 0,
    val completed: Int = 0,
    val failed: Int = 0,
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
 * World-space point reconstructed on the server.
 *
 * @param zoneId zone that owns this reconstructed point.
 * @param position point position in world coordinates.
 * @param confidence point confidence in range `[0, 1]`.
 */
@Serializable
data class ServerWorldPointDto(
    val zoneId: Long,
    @Serializable(with = Vector3Serializer::class)
    val position: Vector3F,
    val confidence: Float
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
) {
    val centerPoint = ImagePoint(
        x = x + width / 2,
        y = y + height / 2,
    )
}

/**
 * Segmented object instance returned by Python service.
 *
 * @param id instance identifier inside one prediction response.
 * @param bbox detected object bounding box.
 * @param confidence model confidence in range `[0, 1]`.
 */
@Serializable
data class SegmentationInstance(
    val id: Int,
    val bbox: SegmentationBoundingBox,
    val confidence: Float,
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
 * @param distortionCoefficients lens distortion coefficients.
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
    val distortionCoefficients: List<Float> = emptyList(),
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
