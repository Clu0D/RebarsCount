package anton.axenov

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.util.IdentityHashMap
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F

/**
 * Stores per-zone snapshots captured from different viewing angles.
 *
 * @param minAngleDifferenceDegrees minimum required spherical angle difference between stored snapshots.
 * @param onSnapshotStored callback invoked when snapshot is added.
 * @param onSnapshotRemoved callback invoked when snapshot is removed.
 */
class SnapshotsManager(
    private val minAngleDifferenceDegrees: Float = 5f,
    private val onSnapshotStored: (zone: Zone, snapshot: ZoneSnapshot) -> Unit = { _, _ -> },
    private val onSnapshotRemoved: (zone: Zone, snapshot: ZoneSnapshot) -> Unit = { _, _ -> },
) {
    private val snapshotsByZone = IdentityHashMap<Zone, MutableList<ZoneSnapshot>>()

    /**
     * Tries to add one zone snapshot.
     *
     * New snapshot is stored when:
     * 1) snapshot quality is good enough, and
     * 2) there is no existing snapshot within [minAngleDifferenceDegrees], or
     * 3) there is a nearby snapshot but this one has better coverage.
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
        val similarSnapshotIndices = zoneSnapshots
            .indices
            .filter { index ->
                captureAngle.sphericalAngleTo(
                    zoneSnapshots[index].captureAngle,
                ) <= minAngleDifferenceDegrees
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
        return captureAngle.angleDegrees < 120f &&
                screenCoverage.isFullyInside
    }

    /**
     * Returns stored snapshots for one zone.
     */
    fun getZoneSnapshots(zone: Zone): List<ZoneSnapshot> {
        return snapshotsByZone[zone]?.toList().orEmpty()
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
