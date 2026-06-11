package anton.axenov

import kotlin.math.PI
import kotlin.math.acos
import kotlin.math.roundToInt
import korlibs.math.geom.Vector3F as Vector3

/**
 * Calculates polygon area and visible area using target-platform geometry library.
 *
 * @param screenPolygon polygon points in screen pixels.
 * @param screenWidth screen width in pixels.
 * @param screenHeight screen height in pixels.
 * @return geometry-based coverage values.
 */
internal expect fun calculateScreenPolygonCoverage(
    screenPolygon: List<ImagePoint>,
    screenWidth: Int,
    screenHeight: Int,
): ZoneScreenCoverageMetrics

/**
 * Estimates screenshot capture angle for one zone plane.
 *
 * @param planePose zone plane.
 * @param cameraPosition camera position in world coordinates.
 * @return capture-angle metrics.
 */
fun getZoneCaptureAngle(
    planePose: PlanePose,
    cameraPosition: Vector3,
): ZoneCaptureAngle {
    val normalizedNormal = planePose.normal.normalized()
    val zoneToCameraDirection = (cameraPosition - planePose.center).normalized()
    val dot = normalizedNormal.dot(zoneToCameraDirection).coerceIn(-1f, 1f)
    val angleDegrees = (acos(dot.toDouble()) * 180.0 / PI).toFloat()

    val projectedOnPlane = (zoneToCameraDirection - normalizedNormal * dot)
    val normalizedProjected = projectedOnPlane.normalized()

    val worldUp = Vector3(0f, 1f, 0f)
    val axisX = worldUp.cross(normalizedNormal).normalized()
    val axisY = normalizedNormal.cross(axisX).normalized()

    return ZoneCaptureAngle(
        angleDegrees = angleDegrees,
        zoneToCameraDirection = zoneToCameraDirection,
        normalToCameraDot = dot,
        planarDirectionX = normalizedProjected.dot(axisX),
        planarDirectionY = normalizedProjected.dot(axisY),
    )
}

/**
 * Builds one short multiline text for zone metric label.
 *
 * @param zone zone that should be described.
 * @param cameraPosition current camera position in world coordinates.
 * @param screenWidth current screen width in pixels.
 * @param screenHeight current screen height in pixels.
 * @param worldPointProjector projects world point into current screen coordinates (null if can't).
 * @return human-readable metric label.
 */
fun buildZoneMetricsText(
    zone: Zone,
    cameraPosition: Vector3,
    screenWidth: Int,
    screenHeight: Int,
    worldPointProjector: (Vector3) -> ViewPoint?,
): String {
    val captureAngle = getZoneCaptureAngle(
        planePose = zone.planePose,
        cameraPosition = cameraPosition,
    )
    val coverage = getZoneScreenCoverage(
        zone = zone,
        screenWidth = screenWidth,
        screenHeight = screenHeight,
        worldPointProjector = worldPointProjector,
    )
    val mergeText = if (zone.isPlaced())
        "placed"
    else
        (zone.mergeLabelText ?: "no merge") + " insignificantChanges=${zone.insignificantChanges}"

    return "id=${zone.id}, \nang=${captureAngle.angleDegrees.toPrecision(1)}deg, " +
            "dot=${captureAngle.normalToCameraDot.toPrecision(2)}\n" +
            "dir2d=(${captureAngle.planarDirectionX.toPrecision(2)}," +
            "${captureAngle.planarDirectionY.toPrecision(2)}) " +
            "cov=${(coverage.coverage * 100f).toPrecision(1)}%, " +
            "in=${if (coverage.isFullyInside) "Y" else "N"}\n" +
            mergeText + "\n" +
            "points=${zone.sceneWorldPointsCount}, "
}

/**
 * Projects current zone polygon to current camera view and calculates its screen coverage.
 *
 * @param zone zone to project.
 * @param screenWidth current screen width.
 * @param screenHeight current screen height.
 * @param worldPointProjector projects world point into current screen coordinates (null if can't).
 * @return current screen coverage for zone polygon.
 */
fun getZoneScreenCoverage(
    zone: Zone,
    screenWidth: Int,
    screenHeight: Int,
    worldPointProjector: (Vector3) -> ViewPoint?,
): ZoneScreenCoverageMetrics {
    val screenArea = screenWidth.toFloat() * screenHeight.toFloat()
    if (zone.polygonPoints.size < 3 || screenWidth <= 0 || screenHeight <= 0) {
        return ZoneScreenCoverageMetrics(
            projectedArea = 0f,
            visibleArea = 0f,
            isFullyInside = false,
            screenArea = screenArea,
        )
    }

    val projectedPolygon = projectZonePolygonToScreen(zone, worldPointProjector)
        ?: return ZoneScreenCoverageMetrics(
            projectedArea = 0f,
            visibleArea = 0f,
            isFullyInside = false,
            screenArea = screenArea,
        )
    return calculateScreenPolygonCoverage(
        screenPolygon = projectedPolygon,
        screenWidth = screenWidth,
        screenHeight = screenHeight,
    )
}

/**
 * Projects current zone polygon to current camera view.
 *
 * @param zone zone to project.
 * @param worldPointProjector projects world point into current screen coordinates (null if can't).
 * @return projected polygon in screen pixels or null when projection failed.
 */
fun projectZonePolygonToScreen(
    zone: Zone,
    worldPointProjector: (Vector3) -> ViewPoint?,
): List<ImagePoint>? {
    if (zone.polygonPoints.size < 3) {
        return null
    }
    return zone.polygonPoints.map { worldPoint ->
        val projected = worldPointProjector(worldPoint) ?: return null
        ImagePoint(
            x = projected.xPx.roundToInt(),
            y = projected.yPx.roundToInt(),
        )
    }
}
