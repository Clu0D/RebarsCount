package anton.axenov

import korlibs.math.geom.Vector3F as Vector3

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
)

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
}
