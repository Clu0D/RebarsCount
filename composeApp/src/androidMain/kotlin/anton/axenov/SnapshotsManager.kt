package anton.axenov

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.util.IdentityHashMap
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F
import kotlin.math.sqrt

/**
 * Stores per-zone snapshots captured from different viewing angles.
 *
 * @param onSnapshotStored callback invoked when snapshot is added.
 * @param onSnapshotRemoved callback invoked when snapshot is removed.
 */
class SnapshotsManager(
    private val onSnapshotStored: (zone: Zone, snapshot: ZoneSnapshot) -> Unit = { _, _ -> },
    private val onSnapshotRemoved: (zone: Zone, snapshot: ZoneSnapshot) -> Unit = { _, _ -> },
) {
    private val snapshotsByZone = IdentityHashMap<Zone, MutableList<ZoneSnapshot>>()

    /**
     * Tries to add one zone snapshot.
     *
     * New snapshot is stored when:
     * 1) snapshot quality is good enough, and
     * 2) there is no existing snapshot with `Delta_combined <= 1`, or
     * 3) there is a similar snapshot but this one has better coverage.
     *
     * @param zone zone owner.
     * @param frameSnapshot current camera frame snapshot.
     * @param captureAngle angle metrics for this zone at the current frame.
     * @param screenCoverage coverage metrics for this zone at the current frame.
     * @return add decision details.
     */
    fun addSnapshot(
        zone: Zone,
        frameSnapshot: DetectionFrameSnapshot,
        captureAngle: ZoneCaptureAngle,
        screenCoverage: ZoneScreenCoverageMetrics,
    ): SnapshotStoreDecision {
        if (!isSnapshotGood(captureAngle, screenCoverage)) {
            return SnapshotStoreDecision.SKIPPED_BAD_QUALITY
        }
        val zoneSnapshots = snapshotsByZone.getOrPut(zone) { mutableListOf() }
        val currentCameraPosition = frameSnapshot.cameraWorldPosition()
        val currentDistanceMeters = distanceMeters(currentCameraPosition, zone.planePose.center)
        val similarSnapshotIndices = zoneSnapshots.indices.filter { index ->
            val storedSnapshot = zoneSnapshots[index]
            val similarityMetrics = calculateSnapshotSimilarityMetrics(
                zone = zone,
                firstCaptureAngle = captureAngle,
                firstCameraPosition = currentCameraPosition,
                firstDistanceMeters = currentDistanceMeters,
                secondSnapshot = storedSnapshot,
            )
            similarityMetrics.deltaCombined <= MAX_SIMILAR_SNAPSHOT_DELTA_COMBINED
        }

        if (similarSnapshotIndices.isEmpty()) {
            val newSnapshot = zoneSnapshotFromFrame(frameSnapshot, captureAngle, screenCoverage)
            zoneSnapshots += newSnapshot
            onSnapshotStored(zone, newSnapshot)
            return SnapshotStoreDecision.ADDED_NEW_ANGLE
        }

        val bestSimilarIndex = similarSnapshotIndices
            .maxByOrNull { index -> zoneSnapshots[index].screenCoverage.coverage }
            ?: return SnapshotStoreDecision.SKIPPED_LOWER_COVERAGE
        val existingCoverage = zoneSnapshots[bestSimilarIndex].screenCoverage.coverage
        val newCoverage = screenCoverage.coverage
        if (newCoverage <= existingCoverage) {
            return SnapshotStoreDecision.SKIPPED_LOWER_COVERAGE
        }

        val replacement = zoneSnapshotFromFrame(frameSnapshot, captureAngle, screenCoverage)
        similarSnapshotIndices
            .sortedDescending()
            .forEach { index ->
                val removedSnapshot = zoneSnapshots[index]
                removedSnapshot.frameSnapshot.screenshot.recycle()
                onSnapshotRemoved(zone, removedSnapshot)
                zoneSnapshots.removeAt(index)
            }
        zoneSnapshots += replacement
        onSnapshotStored(zone, replacement)
        return SnapshotStoreDecision.REPLACED_BETTER_COVERAGE
    }

    /**
     * Checks whether snapshot quality is acceptable for storage.
     */
    fun isSnapshotGood(
        captureAngle: ZoneCaptureAngle,
        screenCoverage: ZoneScreenCoverageMetrics
    ): Boolean {
        val theta = captureAngle.angleDegrees
        val gamma = screenCoverage.coverage
        return theta < MAX_SNAPSHOT_CAPTURE_ANGLE_DEGREES &&
            gamma > MIN_SCREEN_COVERAGE_RATIO
    }

    /**
     * Returns stored snapshots for one zone.
     */
    fun getZoneSnapshots(zone: Zone): List<ZoneSnapshot> {
        return snapshotsByZone[zone]?.toList().orEmpty()
    }

    /**
     * Returns payloads for every currently stored snapshot with its owning zone.
     *
     * @return current snapshot payloads in zone iteration order.
     */
    fun getAllSnapshotPayloads(): List<ZoneSnapshotUploadDto> {
        return snapshotsByZone.flatMap { (zone, snapshots) ->
            snapshots.map { snapshot -> snapshot.toPayload(zone) }
        }
    }

    /**
     * Returns total number of currently stored snapshots.
     *
     * @return stored snapshot count.
     */
    fun getSnapshotCount(): Int {
        return snapshotsByZone.values.sumOf { snapshots -> snapshots.size }
    }

    /**
     * Removes all snapshots for one zone and frees bitmap memory.
     *
     * @return removed snapshot count.
     */
    fun removeZone(zone: Zone): Int {
        val removed = snapshotsByZone.remove(zone) ?: return 0
        removed.forEach { snapshot ->
            snapshot.frameSnapshot.screenshot.recycle()
            onSnapshotRemoved(zone, snapshot)
        }
        return removed.size
    }

    /**
     * Removes all stored zone snapshots and frees bitmap memory.
     */
    fun clear() {
        snapshotsByZone.forEach { (zone, zoneSnapshots) ->
            zoneSnapshots.forEach { snapshot ->
                snapshot.frameSnapshot.screenshot.recycle()
                onSnapshotRemoved(zone, snapshot)
            }
        }
        snapshotsByZone.clear()
    }

    /**
     * Builds persisted zone snapshot payload by copying frame image and metadata.
     *
     * @param frameSnapshot source frame snapshot.
     * @param captureAngle angle metrics for this zone at the current frame.
     * @param screenCoverage coverage metrics for this zone at the current frame.
     * @return persisted zone snapshot payload.
     */
    private fun zoneSnapshotFromFrame(
        frameSnapshot: DetectionFrameSnapshot,
        captureAngle: ZoneCaptureAngle,
        screenCoverage: ZoneScreenCoverageMetrics,
    ): ZoneSnapshot {
        return ZoneSnapshot(
            frameSnapshot.copy(
                screenshot = frameSnapshot.screenshot.copyBitmapForStorage(),
            ),
            captureAngle,
            screenCoverage
        )
    }
}

/**
 * Snapshot storing decision returned by [SnapshotsManager].
 */
enum class SnapshotStoreDecision {
    ADDED_NEW_ANGLE,
    REPLACED_BETTER_COVERAGE,
    SKIPPED_LOWER_COVERAGE,
    SKIPPED_BAD_QUALITY,
}

/**
 * Persisted screenshot payload for one zone.
 *
 * @param frameSnapshot copied frame snapshot with camera image and intrinsics.
 * @param captureAngle angle metrics for this zone at the current frame.
 * @param screenCoverage coverage metrics for this zone at the current frame.
 */
data class ZoneSnapshot(
    val frameSnapshot: DetectionFrameSnapshot,
    val captureAngle: ZoneCaptureAngle,
    val screenCoverage: ZoneScreenCoverageMetrics,
) {
    fun toPayload(zone: Zone): ZoneSnapshotUploadDto {
        return ZoneSnapshotUploadDto(
            zone = zone,
            frameSnapshot = frameSnapshot.toPayload(),
            captureAngle = captureAngle,
            screenCoverage = screenCoverage,
        )
    }
}

fun DetectionFrameSnapshot.toPayload(): DetectionFrameSnapshotDto {
    return DetectionFrameSnapshotDto(
        screenshotPngBytes = screenshot.toPngByteArray(),
        frameTimestamp = frameTimestamp,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        focalLengthX = focalLengthX,
        focalLengthY = focalLengthY,
        principalPointX = principalPointX,
        principalPointY = principalPointY,
        distortionCoefficients = distortionCoefficients,
        cameraPose = CameraPoseDto(
            translation = Vector3F(
                x = cameraPose.tx(),
                y = cameraPose.ty(),
                z = cameraPose.tz(),
            ),
            rotationQuaternion = Quaternion(
                x = cameraPose.qx(),
                y = cameraPose.qy(),
                z = cameraPose.qz(),
                w = cameraPose.qw(),
            ),
        ),
        depthSnapshot = depthSnapshot,
    )
}

fun Bitmap.toPngByteArray(): ByteArray {
    val outputStream = ByteArrayOutputStream()
    compress(Bitmap.CompressFormat.PNG, 100, outputStream)
    return outputStream.toByteArray()
}

/**
 * Calculates snapshot similarity metrics
 *
 * @param zone zone shared by both snapshots.
 * @param firstCaptureAngle capture-angle metrics of the current snapshot.
 * @param firstCameraPosition current camera position.
 * @param firstDistanceMeters current camera-to-zone distance.
 * @param secondSnapshot stored snapshot to compare against.
 * @return combined angular/planar difference metrics.
 */
private fun calculateSnapshotSimilarityMetrics(
    zone: Zone,
    firstCaptureAngle: ZoneCaptureAngle,
    firstCameraPosition: Vector3F,
    firstDistanceMeters: Float,
    secondSnapshot: ZoneSnapshot,
): SnapshotSimilarityMetrics {
    val secondCameraPosition = secondSnapshot.frameSnapshot.cameraWorldPosition()
    val secondDistanceMeters = distanceMeters(secondCameraPosition, zone.planePose.center)
    val delta = firstCaptureAngle.sphericalAngleTo(secondSnapshot.captureAngle)
    val deltaLinear = calculatePlanarCameraShiftMeters(
        firstCameraPosition = firstCameraPosition,
        secondCameraPosition = secondCameraPosition,
        planeNormal = zone.planePose.normal,
    )
    val distanceForNormalization = ((firstDistanceMeters + secondDistanceMeters) / 2f)
        .coerceAtLeast(MIN_CAMERA_DISTANCE_METERS)
    val deltaNormalized = deltaLinear / distanceForNormalization
    val deltaCombined =
        (delta / MIN_SPHERICAL_DIFFERENCE_DEGREES) +
            (deltaNormalized / MIN_NORMALIZED_PLANAR_SHIFT)
    return SnapshotSimilarityMetrics(
        delta = delta,
        deltaLinear = deltaLinear,
        deltaNormalized = deltaNormalized,
        deltaCombined = deltaCombined,
    )
}

/**
 * Calculates planar camera shift `Delta` between 2 camera positions for one zone plane.
 *
 * @param firstCameraPosition first camera position in world coordinates.
 * @param secondCameraPosition second camera position in world coordinates.
 * @param planeNormal zone plane normal.
 * @return linear shift component lying along the plane.
 */
private fun calculatePlanarCameraShiftMeters(
    firstCameraPosition: Vector3F,
    secondCameraPosition: Vector3F,
    planeNormal: Vector3F,
): Float {
    val positionDelta = firstCameraPosition - secondCameraPosition
    val deltaSquared = positionDelta.dot(positionDelta)
    val normalComponent = planeNormal.normalized().dot(positionDelta)
    val planarSquared = (deltaSquared - normalComponent * normalComponent).coerceAtLeast(0f)
    return sqrt(planarSquared)
}

/**
 * Calculates Euclidean distance between 2 world points.
 *
 * @param firstPoint first point.
 * @param secondPoint second point.
 * @return distance in meters.
 */
private fun distanceMeters(
    firstPoint: Vector3F,
    secondPoint: Vector3F,
): Float = (firstPoint - secondPoint).length

/**
 * Extracts camera position from frame snapshot pose.
 *
 * @return camera world position.
 */
private fun DetectionFrameSnapshot.cameraWorldPosition(): Vector3F {
    return Vector3F(
        x = cameraPose.tx(),
        y = cameraPose.ty(),
        z = cameraPose.tz(),
    )
}

/**
 * Combined similarity metrics between 2 snapshots of the same zone.
 *
 * @param delta spherical angle difference in degrees.
 * @param deltaLinear planar camera shift `Delta` in meters.
 * @param deltaNormalized normalized planar shift `Delta_n`.
 * @param deltaCombined combined metric.
 */
private data class SnapshotSimilarityMetrics(
    val delta: Float,
    val deltaLinear: Float,
    val deltaNormalized: Float,
    val deltaCombined: Float,
)

private const val MAX_SNAPSHOT_CAPTURE_ANGLE_DEGREES = 75f
private const val MIN_SCREEN_COVERAGE_RATIO = 0.3f
private const val MIN_SPHERICAL_DIFFERENCE_DEGREES = 15f
private const val MIN_NORMALIZED_PLANAR_SHIFT = 0.1f
private const val MAX_SIMILAR_SNAPSHOT_DELTA_COMBINED = 1f
private const val MIN_CAMERA_DISTANCE_METERS = 0.01f
