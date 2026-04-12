package anton.axenov

import korlibs.math.geom.Vector3F as Vector3
import kotlin.math.max
import kotlin.math.min

/**
 * One detected zone represented in world space.
 *
 * @param id unique zone identifier.
 * @param sampledPoints sampled world points used to estimate infinite plane.
 * @param planePose mathematical parameters of fitted infinite plane.
 * @param projectionInputs all original projection payloads used to build source polygons.
 */
data class Zone(
    val id: Long = nextZoneId(),
    val sampledPoints: List<Vector3>,
    val planePose: PlanePose,
    val projectionInputs: List<ZoneProjectionInput> = emptyList(),
) {
    var metricsLabelText: String = "Metrics: waiting"
    var serverLabelText: String? = null

    /**
     * Current text shown in the zone label in AR scene.
     */
    val labelText: String
        get() = "$metricsLabelText\n${serverLabelText ?: ""}"

    val polygonPoints: List<Vector3> by lazy {
        if (projectionInputs.isEmpty())
            return@lazy emptyList()

        val projectedPoints = projectionInputs
            .flatMap { input -> input.projectToPlane(planePose).orEmpty() }
        if (projectedPoints.isEmpty())
            return@lazy emptyList()

        buildConvexHullOnPlane(
            worldPoints = projectedPoints,
            planePose = planePose,
        )
    }


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

    val center: Vector3 by lazy {
        Vector3(
            (boundingBox.maxX + boundingBox.minX) / 2f,
            (boundingBox.maxY + boundingBox.minY) / 2f,
            (boundingBox.maxZ + boundingBox.minZ) / 2f,
        )
    }
}

/**
 * Full projection payload required to reproject one zone polygon onto any plane.
 *
 * @param originalScreenPolygon detector-space polygon points before translation.
 * @param translationVariant detector-to-image translation variant.
 * @param imageWidth captured image width.
 * @param imageHeight captured image height.
 * @param focalLengthX camera focal length X in pixels.
 * @param focalLengthY camera focal length Y in pixels.
 * @param principalPointX camera principal point X in pixels.
 * @param principalPointY camera principal point Y in pixels.
 * @param cameraToWorldMatrix camera pose matrix in OpenGL column-major form.
 */
class ZoneProjectionInput(
    val originalScreenPolygon: List<ImagePoint>,
    val translationVariant: CoordinateTranslationVariant,
    val imageWidth: Int,
    val imageHeight: Int,
    val focalLengthX: Float,
    val focalLengthY: Float,
    val principalPointX: Float,
    val principalPointY: Float,
    val cameraToWorldMatrix: FloatArray,
) {
    /**
     * Returns camera world position extracted from [cameraToWorldMatrix].
     *
     * @return camera world position.
     */
    fun cameraPosition(): Vector3 {
        return Vector3(
            x = cameraToWorldMatrix[12],
            y = cameraToWorldMatrix[13],
            z = cameraToWorldMatrix[14],
        )
    }

    /**
     * Projects original on-screen polygon onto [planePose].
     *
     * @param planePose projection plane.
     * @return projected world-space polygon or null when at least one corner cannot be projected.
     */
    fun projectToPlane(planePose: PlanePose): List<Vector3>? {
        val translatedCorners = originalScreenPolygon.map { corner ->
            translateCoordinates(
                x = corner.x,
                y = corner.y,
                width = imageWidth,
                height = imageHeight,
                translationVariant = translationVariant,
            )
        }
        val projectedCorners = translatedCorners.map { corner ->
            projectImagePointToPlane(
                imagePoint = corner,
                planePose = planePose,
            )
        }
        if (projectedCorners.any { it == null }) {
            return null
        }
        return projectedCorners.map { it!! }
    }

    /**
     * Projects one image-space point to [planePose].
     *
     * @param imagePoint image-space point to project.
     * @param planePose projection plane.
     * @return world-space intersection or null when ray is parallel to plane or intersects behind camera.
     */
    private fun projectImagePointToPlane(
        imagePoint: ImagePoint,
        planePose: PlanePose,
    ): Vector3? {
        if (
            imageWidth <= 0 ||
            imageHeight <= 0 ||
            focalLengthX == 0f ||
            focalLengthY == 0f
        ) {
            return null
        }
        val imageX = imagePoint.x.coerceIn(0, imageWidth - 1)
        val imageY = imagePoint.y.coerceIn(0, imageHeight - 1)

        val directionCamera = Vector3(
            x = (imageX - principalPointX) / focalLengthX,
            y = -(imageY - principalPointY) / focalLengthY,
            z = -1f,
        ).normalized()
        val matrix = cameraToWorldMatrix
        val rayOrigin = cameraPosition()
        val rayDirection = Vector3(
            x = matrix[0] * directionCamera.x + matrix[4] * directionCamera.y + matrix[8] * directionCamera.z,
            y = matrix[1] * directionCamera.x + matrix[5] * directionCamera.y + matrix[9] * directionCamera.z,
            z = matrix[2] * directionCamera.x + matrix[6] * directionCamera.y + matrix[10] * directionCamera.z,
        ).normalized()

        val denominator = planePose.normal.dot(rayDirection)
        if (kotlin.math.abs(denominator) <= RAY_PLANE_EPSILON) {
            return null
        }
        val distance = planePose.normal.dot(planePose.center - rayOrigin) / denominator
        if (distance <= 0f) {
            return null
        }
        return rayOrigin + rayDirection * distance
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
        val zonesToMerge = listOf(newZone) + removedZones
        val mergedProjectionInputs = zonesToMerge.flatMap { zone -> zone.projectionInputs }
        val mergedSampledPoints = zonesToMerge.flatMap { zone -> zone.sampledPoints }
        val cameraPosition = mergedProjectionInputs.firstOrNull()?.cameraPosition() ?: newZone.planePose.center
        val mergedPlanePose = fitPlanePoseFromPoints(
            worldPoints = mergedSampledPoints,
            cameraPosition = cameraPosition,
            minPointCount = MERGE_PLANE_MIN_POINT_COUNT,
        ).pose ?: newZone.planePose
        return ZoneMergeResult(
            zone = Zone(
                sampledPoints = mergedSampledPoints,
                planePose = mergedPlanePose,
                projectionInputs = mergedProjectionInputs,
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
private const val RAY_PLANE_EPSILON = 1e-5f
private const val MERGE_PLANE_MIN_POINT_COUNT = 3

private var nextGeneratedZoneId: Long = 1L
private fun nextZoneId(): Long = nextGeneratedZoneId++