package anton.axenov

import korlibs.math.geom.Vector3F as Vector3

/**
 * Stores all world-space zones.
 *
 * On add, close zones are merged by bounding-box overlap threshold.
 *
 * @param onZoneAddition callback invoked for each zone added to storage.
 * @param onZoneDeletion callback invoked for each zone removed from storage.
 * @param onZoneLabelUpdate callback invoked when stored zone label text changes.
 */
class ZonesManager(
    private val onZoneAddition: (Zone) -> Unit = {},
    private val onZoneDeletion: (Zone) -> Unit = {},
    private val onZoneLabelUpdate: (Zone) -> Unit = {},
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
            val mergeResult = tryMergeZones(newZone)
            val mergedZone = mergeResult.zone ?: return@forEach
            zones += mergedZone
            onZoneAddition(mergedZone)
            mergeDebugInfo += "${mergeResult.overlapZonesCount}: ${mergeResult.maxOverlapPercent}"
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
     * Recalculates and applies zone label metrics for all stored zones.
     *
     * @param cameraPosition current camera position in world coordinates.
     * @param screenWidth current screen width in pixels.
     * @param screenHeight current screen height in pixels.
     * @param worldPointProjector projects world points to current screen coordinates (null if can't).
     * @return number of zones whose visible label text changed.
     */
    fun refreshZoneMetricsLabels(
        cameraPosition: Vector3,
        screenWidth: Int,
        screenHeight: Int,
        worldPointProjector: (Vector3) -> ViewPoint?,
    ): Int {
        var changedZonesCount = 0
        zones.forEach { zone ->
            val previousLabelText = zone.labelText
            val nextLabelText = buildZoneMetricsText(
                zone = zone,
                cameraPosition = cameraPosition,
                screenWidth = screenWidth,
                screenHeight = screenHeight,
                worldPointProjector = worldPointProjector,
            )
            zone.metricsLabelText = nextLabelText
            if (previousLabelText != zone.labelText) {
                onZoneLabelUpdate(zone)
                changedZonesCount++
            }
        }
        return changedZonesCount
    }

    /**
     * Applies server texts to currently stored zones by identifier.
     *
     * @param textsByZoneId map of zone id to server text.
     * @return number of zones whose combined label text changed.
     */
    fun applyServerTexts(textsByZoneId: Map<Long, String>): Int {
        if (textsByZoneId.isEmpty()) {
            return 0
        }
        var changedZonesCount = 0
        zones.forEach { zone ->
            val nextServerText = textsByZoneId[zone.id] ?: return@forEach
            val previousLabelText = zone.labelText
            zone.serverLabelText = nextServerText
            if (previousLabelText != zone.labelText) {
                onZoneLabelUpdate(zone)
                changedZonesCount++
            }
        }
        return changedZonesCount
    }

    /**
     * Applies current server world-point counts to stored zones by identifier.
     *
     * @param worldPointCountsByZoneId map of zone id to number of reconstructed world points in scene.
     * @return number of zones whose visible label text changed.
     */
    fun applyWorldPointCounts(worldPointCountsByZoneId: Map<Long, Int>): Int {
        var changedZonesCount = 0
        zones.forEach { zone ->
            val previousLabelText = zone.labelText
            zone.sceneWorldPointsCount = worldPointCountsByZoneId[zone.id] ?: 0
            if (previousLabelText != zone.labelText) {
                onZoneLabelUpdate(zone)
                changedZonesCount++
            }
        }
        return changedZonesCount
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
     * Returns immutable snapshot of all currently stored zones.
     *
     * @return stored zones in insertion order.
     */
    fun getZones(): List<Zone> = zones.toList()

    /**
     * Returns immutable snapshot of all zones that are already placed in the world.
     *
     * @return placed zones in insertion order.
     */
    fun getPlacedZones(): List<Zone> = zones.filter { zone -> zone.isPlaced() }

    /**
     * Try merging newly added zone with old ones.
     *
     * Merged with stored zone with the biggest significant overlap.
     * Old zone is removed from scene if merged.
     * If no zones to merge new one is added to the scene.
     *
     * @param newZone newly detected zone.
     * @return new zone to add to the scene or null.
     */
    private fun tryMergeZones(newZone: Zone): ZoneMergeResult {
        val overlaps = findZoneOverlaps(newZone)
        val biggestOverlap = overlaps
            .filter { (_, overlap) -> overlap >= BOX_INTERSECTION_THRESHOLD }
            .maxByOrNull { (_, overlap) -> overlap }
        val maxOverlapPercent = ((biggestOverlap?.second) ?: 0f) * 100f

        if (biggestOverlap == null) {
            return ZoneMergeResult(
                zone = newZone,
                overlapZonesCount = 0,
                maxOverlapPercent = maxOverlapPercent,
            )
        }

        val zoneToMerge = biggestOverlap.first
        if (zoneToMerge.isPlaced()) {
            return ZoneMergeResult(
                zone = null,
                overlapZonesCount = 0,
                maxOverlapPercent = maxOverlapPercent,
            )
        }

        val removedZones = removeZones(listOf(zoneToMerge))
        if (removedZones.isNotEmpty())
            queuedZonesToRemove += removedZones

        val zonesToMerge = removedZones + newZone
        val mergedProjectionInputs = zonesToMerge.flatMap { zone -> zone.projectionInputs }
        val mergedSampledPoints = zonesToMerge.flatMap { zone -> zone.sampledPoints }
        val cameraPosition = mergedProjectionInputs.firstOrNull()?.cameraPosition() ?: newZone.planePose.center
        val mergedPlanePose = fitPlanePoseFromPoints(
            worldPoints = mergedSampledPoints,
            cameraPosition = cameraPosition,
            minPointCount = MERGE_PLANE_MIN_POINT_COUNT,
        ).pose ?: newZone.planePose
        val mergedZone = Zone(
            sampledPoints = mergedSampledPoints,
            planePose = mergedPlanePose,
            projectionInputs = mergedProjectionInputs,
        )
        val zoneChangeInsignificance = wasZoneChangeInsignificant(mergedZone, zonesToMerge)
        mergedZone.mergeLabelText = zoneChangeInsignificance.mergeLabelText
        mergedZone.insignificantChanges = if (zoneChangeInsignificance.wasChangeInsignificant)
            zoneToMerge.insignificantChanges + 1
        else
            0
        return ZoneMergeResult(
            zone = mergedZone,
            overlapZonesCount = removedZones.size,
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
 * @param overlapZonesCount merge statistics.
 * @param maxOverlapPercent merge statistics.
 */
private data class ZoneMergeResult(
    val zone: Zone?,
    val overlapZonesCount: Int,
    val maxOverlapPercent: Float,
)

private const val BOX_INTERSECTION_THRESHOLD = 0.5f
private const val MERGE_PLANE_MIN_POINT_COUNT = 3
