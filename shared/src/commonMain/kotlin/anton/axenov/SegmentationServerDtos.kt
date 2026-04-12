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
data class ServerHealthResponseDto(
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
data class ZoneStatusDto(
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
data class SnapshotUploadResponseDto(
    val ok: Boolean,
    val zoneId: Long,
    val snapshotCount: Int,
    val message: String,
)

/**
 * Snapshot for one zone.
 *
 * @param zoneId requested zone identifier.
 * @param snapshotCount number of stored snapshots.
 * @param snapshots stored snapshot payloads.
 * @param text current zone status text.
 */
@Serializable
data class ZoneSnapshotsResponseDto(
    val zoneId: Long,
    val snapshotCount: Int,
    val snapshots: List<ZoneSnapshotUploadDto>,
    val text: String,
)

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
