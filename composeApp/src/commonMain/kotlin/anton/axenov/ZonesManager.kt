package anton.axenov

import korlibs.math.geom.Vector3F as Vector3
import kotlin.math.max
import kotlin.math.min

/**
 * One detected zone represented in world space.
 *
 * @param polygonPoints projected zone polygon points in world coordinates.
 * @param sampledPoints sampled world points used to estimate infinite plane.
 * @param planePose mathematical parameters of fitted infinite plane.
 */
data class Zone(
    val polygonPoints: List<Vector3>,
    val sampledPoints: List<Vector3>,
    val planePose: PlanePose,
) {
    val boundingBox: ZoneBoundingBox3d by lazy {
        val basePoints = when {
            polygonPoints.isNotEmpty() -> polygonPoints
            sampledPoints.isNotEmpty() -> sampledPoints
            else -> listOf(planePose.center)
        }

        val minX = basePoints.minOf { it.x }
        val minY = basePoints.minOf { it.y }
        val minZ = basePoints.minOf { it.z }
        val maxX = basePoints.maxOf { it.x }
        val maxY = basePoints.maxOf { it.y }
        val maxZ = basePoints.maxOf { it.z }

        val sizeX = maxX - minX
        val sizeY = maxY - minY
        val sizeZ = maxZ - minZ
        val maxSize = max(max(sizeX, sizeY), sizeZ)
        val padding = max(maxSize * BOUNDING_BOX_PADDING_RATIO, MIN_PADDING_METERS)

        ZoneBoundingBox3d(
            minX = minX - padding,
            minY = minY - padding,
            minZ = minZ - padding,
            maxX = maxX + padding,
            maxY = maxY + padding,
            maxZ = maxZ + padding,
        )
    }
}

/**
 * Axis-aligned 3D bounds used to compare zone overlap.
 *
 * @param minX minimum X coordinate.
 * @param minY minimum Y coordinate.
 * @param minZ minimum Z coordinate.
 * @param maxX maximum X coordinate.
 * @param maxY maximum Y coordinate.
 * @param maxZ maximum Z coordinate.
 */
data class ZoneBoundingBox3d(
    val minX: Float,
    val minY: Float,
    val minZ: Float,
    val maxX: Float,
    val maxY: Float,
    val maxZ: Float,
) {
    /**
     * Computes this box volume.
     *
     * @return positive box volume.
     */
    fun volume(): Float {
        return (maxX - minX).coerceAtLeast(0f) *
                (maxY - minY).coerceAtLeast(0f) *
                (maxZ - minZ).coerceAtLeast(0f)
    }

    /**
     * Computes intersection volume with another box.
     *
     * @param other second box.
     * @return intersection volume or zero when there is no overlap.
     */
    fun intersectionVolume(other: ZoneBoundingBox3d): Float {
        val overlapX = min(maxX, other.maxX) - max(minX, other.minX)
        val overlapY = min(maxY, other.maxY) - max(minY, other.minY)
        val overlapZ = min(maxZ, other.maxZ) - max(minZ, other.minZ)
        if (overlapX <= 0f || overlapY <= 0f || overlapZ <= 0f) {
            return 0f
        }
        return overlapX * overlapY * overlapZ
    }

    /**
     * Computes overlap ratio relative to the smaller box volume.
     *
     * @param other second box.
     * @return value in `[0, 1]` where `0` means no overlap.
     */
    fun overlapRatioBySmallerBox(other: ZoneBoundingBox3d): Float {
        val intersection = intersectionVolume(other)
        if (intersection <= 0f) {
            return 0f
        }
        val denominator = min(volume(), other.volume()).coerceAtLeast(MIN_BOX_VOLUME_EPSILON)
        return intersection / denominator
    }
}

/**
 * Stores all world-space zones and exposes only zones that were not drawn yet.
 *
 * Current behavior only appends zones. A later step can merge nearby zones.
 */
class ZonesManager {
    private val zones = mutableListOf<Zone>()
    private val queuedZonesToRemove = mutableListOf<Zone>()
    private var consumedZoneCount = 0

    /**
     * Adds new zones to manager storage.
     *
     * @param newZones zones to append.
     */
    fun addZones(newZones: List<Zone>) {
        zones.addAll(newZones)
    }

    /**
     * Adds zones to removal queue.
     *
     * Queued zones are removed when [consumeQueuedRemovedZones] is called.
     *
     * @param zonesToRemove zones that should be removed.
     */
    fun addZonesToRemove(zonesToRemove: List<Zone>) {
        if (zonesToRemove.isEmpty()) {
            return
        }
        queuedZonesToRemove += zonesToRemove
    }

    /**
     * Returns zones that were added since previous consumption call.
     *
     * Returned zones are marked as consumed and will not be returned again.
     *
     * @return list of newly added zones that are not consumed yet.
     */
    fun consumeUndrawnZones(): List<Zone> {
        if (consumedZoneCount >= zones.size) {
            return emptyList()
        }
        val undrawnZones = zones.subList(consumedZoneCount, zones.size).toList()
        consumedZoneCount = zones.size
        return undrawnZones
    }

    /**
     * Removes requested zones from manager storage.
     *
     * @param zonesToRemove zones that should be removed.
     * @return actually removed zones that existed in manager.
     */
    fun removeZones(zonesToRemove: List<Zone>): List<Zone> {
        if (zonesToRemove.isEmpty() || zones.isEmpty()) {
            return emptyList()
        }

        val removeSet = zonesToRemove.toSet()
        val removedZones = mutableListOf<Zone>()
        var removedFromConsumedPrefix = 0
        var originalIndex = 0
        val iterator = zones.listIterator()
        while (iterator.hasNext()) {
            val zone = iterator.next()
            if (zone in removeSet) {
                iterator.remove()
                removedZones += zone
                if (originalIndex < consumedZoneCount) {
                    removedFromConsumedPrefix++
                }
            }
            originalIndex++
        }
        consumedZoneCount = (consumedZoneCount - removedFromConsumedPrefix).coerceAtLeast(0)
        return removedZones
    }

    /**
     * Removes all currently queued zones and clears removal queue.
     *
     * @return actually removed zones that existed in manager.
     */
    fun consumeQueuedRemovedZones(): List<Zone> {
        if (queuedZonesToRemove.isEmpty()) {
            return emptyList()
        }
        val removedZones = removeZones(queuedZonesToRemove)
        queuedZonesToRemove.clear()
        return removedZones
    }

    /**
     * Clears all stored zones and resets consumption state.
     */
    fun clear() {
        zones.clear()
        queuedZonesToRemove.clear()
        consumedZoneCount = 0
    }

    /**
     * Computes overlap ratio between one zone and all stored zones.
     *
     * @param zone zone to compare against stored ones.
     * @return pairs of `(storedZone, overlapRatio)`.
     */
    private fun findZoneOverlaps(zone: Zone): List<Pair<Zone, Float>> {
        val zoneBoundingBox = zone.boundingBox
        return zones
            .asSequence()
            .filter { storedZone -> storedZone !== zone }
            .map { storedZone ->
                storedZone to zoneBoundingBox.overlapRatioBySmallerBox(storedZone.boundingBox)
            }
            .toList()
    }
}

/**
 * One zone merge result.
 *
 * @param zone zone that should be stored after merge.
 * @param intersectingZonesCount merge statistics.
 * @param maxOverlapPercent merge statistics.
 */
private data class ZoneMergeResult(
    val zone: Zone,
    val intersectingZonesCount: Int,
    val maxOverlapPercent: Float,
)

private const val BOUNDING_BOX_PADDING_RATIO = 0.1f
private const val MIN_PADDING_METERS = 0.05f
private const val MIN_BOX_VOLUME_EPSILON = 1e-8f
