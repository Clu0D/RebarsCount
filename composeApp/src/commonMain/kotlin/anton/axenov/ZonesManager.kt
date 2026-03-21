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
     * Clears all stored zones and resets consumption state.
     */
    fun clear() {
        zones.clear()
        consumedZoneCount = 0
    }
}
