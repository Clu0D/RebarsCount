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
 * Stores all world-space zones.
 *
 * On add, close zones are merged by bounding-box overlap threshold.
 *
 * @param onZoneAddition callback invoked for each zone added to storage.
 * @param onZoneDeletion callback invoked for each zone removed from storage.
 */
class ZonesManager(
    private val onZoneAddition: (Zone) -> Unit = {},
    private val onZoneDeletion: (Zone) -> Unit = {},
) {
    private val zones = mutableListOf<Zone>()
    private val queuedZonesToRemove = mutableListOf<Zone>()
    private var mergeDebugInfo: String = ""

    /**
     * Adds new zones to manager storage.
     *
     * Every new zone is compared with current zones by 3D bounding-box overlap.
     * Intersecting old zones are removed and queued for scene removal, then one merged zone is stored.
     *
     * @param newZones zones to append and optionally merge.
     */
    fun addZones(newZones: List<Zone>) {
        newZones.forEach { newZone ->
            val mergeResult = mergeZoneWithIntersectingZones(newZone)
            val mergedZone = mergeResult.zone
            zones += mergedZone
            onZoneAddition(mergedZone)
            mergeDebugInfo += "${mergeResult.intersectingZonesCount}: ${mergeResult.maxOverlapPercent}"
        }
    }

    /**
     * Returns merge diagnostics accumulated during latest [addZones] calls and clears the queue.
     *
     * @return merge diagnostics in insertion order.
     */
    fun consumeMergeDebugInfos(): String =
        mergeDebugInfo.also { mergeDebugInfo = "" }

    /**
     * Finds all currently stored zones that intersect with the provided zone.
     *
     * Two zones are considered intersecting when their 3D bounding boxes overlap
     * by more than BOX_INTERSECTION_THRESHOLD of the smaller box volume.
     *
     * @param zone zone to compare against stored ones.
     * @return intersecting stored zones.
     */
    fun findIntersectingZones(zone: Zone): List<Zone> {
        return findZoneOverlaps(zone)
            .filter { (_, overlap) -> overlap >= BOX_INTERSECTION_THRESHOLD }
            .map { (storedZone, _) -> storedZone }
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
        val iterator = zones.listIterator()
        while (iterator.hasNext()) {
            val zone = iterator.next()
            if (zone in removeSet) {
                iterator.remove()
                removedZones += zone
            }
        }
        removedZones.forEach { zone ->
            onZoneDeletion(zone)
        }
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
     * Clears all stored and queued zones.
     */
    fun clear() {
        val removedZones = zones.toList()
        zones.clear()
        queuedZonesToRemove.clear()
        mergeDebugInfo = ""
        removedZones.forEach { zone ->
            onZoneDeletion(zone)
        }
    }

    /**
     * Merges one newly added zone with all intersecting already stored zones.
     *
     * Intersecting zones are removed from storage and queued for scene removal.
     *
     * @param newZone newly detected zone.
     * @return merged zone containing sampled points from all intersecting zones.
     */
    private fun mergeZoneWithIntersectingZones(newZone: Zone): ZoneMergeResult {
        val overlaps = findZoneOverlaps(newZone)
        val intersectingOverlaps = overlaps
            .filter { (_, overlap) -> overlap >= BOX_INTERSECTION_THRESHOLD }
        val intersectingZones = intersectingOverlaps.map { (storedZone, _) -> storedZone }
        val maxOverlapPercent = (intersectingOverlaps.maxOfOrNull { (_, overlap) -> overlap } ?: 0f) * 100f

        if (intersectingZones.isEmpty()) {
            return ZoneMergeResult(
                zone = newZone,
                intersectingZonesCount = 0,
                maxOverlapPercent = maxOverlapPercent,
            )
        }
        val removedZones = removeZones(intersectingZones)
        if (removedZones.isNotEmpty()) {
            queuedZonesToRemove += removedZones
        }
        return ZoneMergeResult(
            zone = newZone.copy(
                sampledPoints = newZone.sampledPoints + removedZones.flatMap { it.sampledPoints },
            ),
            intersectingZonesCount = removedZones.size,
            maxOverlapPercent = maxOverlapPercent,
        )
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
private const val BOX_INTERSECTION_THRESHOLD = 0.3f
private const val MIN_PADDING_METERS = 0.05f
private const val MIN_BOX_VOLUME_EPSILON = 1e-8f
