package anton.axenov

import kotlin.math.PI
import kotlin.math.acos
import korlibs.math.geom.Vector3F as Vector3

/**
 * Camera-to-zone orientation metrics for one screenshot.
 *
 * @param angleDegrees angle between zone normal and zone-to-camera direction in degrees.
 *      0 is aligned with normal
 *      90 is sideview
 *      > 90 behind the plane
 * @param zoneToCameraDirection normalized direction from plane center to camera position.
 * @param normalToCameraDot dot product of normalized plane normal and zone-to-camera direction.
 */
data class ZoneCaptureAngle(
    val angleDegrees: Float,
    val zoneToCameraDirection: Vector3,
    val normalToCameraDot: Float,
)

/**
 * Estimates screenshot capture angle for one zone plane.
 *
 * @param planePose zone plane.
 * @param cameraPosition camera position in world coordinates.
 * @return capture-angle metrics.
 */
fun estimateZoneCaptureAngle(
    planePose: PlanePose,
    cameraPosition: Vector3,
): ZoneCaptureAngle {
    val normalizedNormal = planePose.normal.normalized()
    val zoneToCameraDirection = (cameraPosition - planePose.center).normalized()
    val dot = normalizedNormal.dot(zoneToCameraDirection).coerceIn(-1f, 1f)
    val angleDegrees = (acos(dot.toDouble()) * 180.0 / PI).toFloat()
    return ZoneCaptureAngle(
        angleDegrees = angleDegrees,
        zoneToCameraDirection = zoneToCameraDirection,
        normalToCameraDot = dot,
    )
}

/**
 * Calculates what percentage of the screen is occupied by one zone bounding box.
 *
 * Bounding-box coordinates are interpreted as inclusive pixel coordinates and are clipped
 * to screen bounds before area calculation.
 *
 * @param screenBoundingBox zone rectangle in screen pixels.
 * @param screenWidth screen width in pixels.
 * @param screenHeight screen height in pixels.
 * @return occupied screen percent in `[0, 100]`.
 */
fun calculateZoneScreenCoveragePercent(
    screenBoundingBox: ScreenBoundingBox,
    screenWidth: Int,
    screenHeight: Int,
): Float {
    if (screenWidth <= 0 || screenHeight <= 0) {
        return 0f
    }

    val clippedLeft = screenBoundingBox.left.coerceIn(0, screenWidth)
    val clippedTop = screenBoundingBox.top.coerceIn(0, screenHeight)
    val clippedRightExclusive = screenBoundingBox.right.coerceIn(0, screenWidth)
    val clippedBottomExclusive = screenBoundingBox.bottom.coerceIn(0, screenHeight)

    val zoneWidth = (clippedRightExclusive - clippedLeft).coerceAtLeast(0)
    val zoneHeight = (clippedBottomExclusive - clippedTop).coerceAtLeast(0)
    val zoneArea = zoneWidth.toLong() * zoneHeight.toLong()
    val screenArea = screenWidth.toLong() * screenHeight.toLong()
    if (screenArea <= 0L) {
        return 0f
    }
    return (zoneArea.toDouble() / screenArea.toDouble() * 100.0).toFloat()
}

/**
 * Builds one short multiline text for zone metric label.
 *
 * @param zone zone that should be described.
 * @param cameraPosition current camera position in world coordinates.
 * @return human-readable metric label.
 */
fun buildZoneMetricsText(
    zone: Zone,
    cameraPosition: Vector3,
): String {
    val captureAngle = estimateZoneCaptureAngle(
        planePose = zone.planePose,
        cameraPosition = cameraPosition,
    )
    val coverages = zone.projectionInputs.map { projectionInput ->
        calculateZoneScreenCoveragePercent(
            screenBoundingBox = ScreenBoundingBox(projectionInput.originalScreenPolygon),
            screenWidth = projectionInput.imageWidth,
            screenHeight = projectionInput.imageHeight,
        )
    }
    val averageCoverage = if (coverages.isEmpty()) 0f else coverages.average().toFloat()
    val maxCoverage = coverages.maxOrNull() ?: 0f

    return "ang=${captureAngle.angleDegrees.toPrecision(1)}deg, " +
            "dot=${captureAngle.normalToCameraDot.toPrecision(2)}\n" +
            "cov(avg/max)=${averageCoverage.toPrecision(1)}%/" +
            "${maxCoverage.toPrecision(1)}% [n=${zone.projectionInputs.size}]"
}
