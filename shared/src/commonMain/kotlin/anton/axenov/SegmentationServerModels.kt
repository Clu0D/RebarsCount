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
 * @param requestId logical request identifier of the uploaded snapshot.
 * @param snapshotCount stored snapshot count for this zone.
 * @param message user-visible server message.
 */
@Serializable
data class SnapshotUploadResponse(
    val ok: Boolean,
    val zoneId: Long,
    val requestId: String,
    val snapshotCount: Int,
    val message: String,
)

/**
 * Request used to delete one queued processing item from the current session.
 *
 * @param requestId logical request identifier to remove.
 */
@Serializable
data class DeleteRequestDto(
    val requestId: String,
)

/**
 * Response returned after deleting one queued processing item.
 *
 * @param ok true when deletion finished successfully.
 * @param requestId logical request identifier that was checked.
 * @param removedSnapshots number of removed queued snapshots.
 * @param message user-visible deletion result.
 */
@Serializable
data class DeleteRequestResponse(
    val ok: Boolean,
    val requestId: String,
    val removedSnapshots: Int,
    val message: String,
)

/**
 * World-space point reconstructed on the server.
 *
 * @param pointId stable point identifier used by post-processing operations.
 * @param zoneId zone that owns this reconstructed point.
 * @param position point position in world coordinates.
 * @param confidence point confidence in range `[0, 1]`.
 */
@Serializable
data class ServerWorldPointDto(
    val pointId: Long,
    val zoneId: Long,
    @Serializable(with = Vector3Serializer::class)
    val position: Vector3F,
    val confidence: Float
)

/**
 * Request used to add one world-space point during post-processing.
 *
 * When [zoneId] is absent, the server assigns the point to the nearest known zone.
 *
 * @param position point position in world coordinates.
 * @param zoneId optional explicit zone identifier.
 * @param confidence point confidence in range `[0, 1]`.
 */
@Serializable
data class AddWorldPointDto(
    @Serializable(with = Vector3Serializer::class)
    val position: Vector3F,
    val zoneId: Long? = null,
    val confidence: Float = 1f,
)

/**
 * Request that identifies one world-space point.
 *
 * @param pointId stable point identifier.
 */
@Serializable
data class WorldPointIdDto(
    val pointId: Long,
)

/**
 * Result of one world-point post-processing operation.
 *
 * @param ok true when the requested mutation was applied.
 * @param point resulting point state, or null when the point was deleted or not found.
 * @param message user-visible mutation result.
 */
@Serializable
data class WorldPointMutationResponse(
    val ok: Boolean,
    val point: ServerWorldPointDto? = null,
    val message: String,
)

/**
 * Request that identifies one zone.
 *
 * @param zoneId stable zone identifier.
 */
@Serializable
data class ZoneIdDto(
    val zoneId: Long,
)

/**
 * Result of cascading zone deletion.
 *
 * @param ok true when the zone existed and was deleted.
 * @param zoneId checked zone identifier.
 * @param removedSnapshots number of removed snapshot records.
 * @param removedQueuedTasks number of removed queued processing tasks.
 * @param removedWorldPoints number of removed automatic and manual world points.
 * @param message user-visible deletion result.
 */
@Serializable
data class DeleteZoneResponse(
    val ok: Boolean,
    val zoneId: Long,
    val removedSnapshots: Int,
    val removedQueuedTasks: Int,
    val removedWorldPoints: Int,
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
 * @param polygon detected object segmentation polygon.
 * @param confidence model confidence in range `[0, 1]`.
 */
@Serializable
data class SegmentationInstance(
    val id: Int,
    val bbox: SegmentationBoundingBox,
    val polygon: List<ImagePoint>,
    val confidence: Float,
) {
    init {
        require(polygon.size >= 3) {
            "Segmentation instance polygon must contain at least three points"
        }
    }

    /**
     * Returns the center of polygon bounds for point-style triangulation.
     */
    val polygonCenterPoint: ImagePoint = ImagePoint(
        x = (polygon.minOf { point -> point.x } + polygon.maxOf { point -> point.x }) / 2,
        y = (polygon.minOf { point -> point.y } + polygon.maxOf { point -> point.y }) / 2,
    )
}

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
 * @param sessionId stable client session identifier.
 * @param requestId logical identifier of this specific snapshot-processing request.
 * @param zone serialized zone payload.
 * @param frameSnapshot serialized frame payload.
 * @param captureAngle serialized capture-angle metrics.
 * @param screenCoverage serialized screen-coverage metrics.
 */
@Serializable
data class ZoneSnapshotUploadDto(
    val sessionId: String,
    val requestId: String,
    val zone: Zone,
    val frameSnapshot: DetectionFrameSnapshotDto,
    val captureAngle: ZoneCaptureAngle,
    val screenCoverage: ZoneScreenCoverageMetrics,
)
