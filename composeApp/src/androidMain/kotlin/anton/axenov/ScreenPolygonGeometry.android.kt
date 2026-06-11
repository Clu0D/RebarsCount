package anton.axenov

import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Envelope
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory


private val zonePolygonGeometryFactory = GeometryFactory()

/**
 * Calculates polygon coverage using JTS geometry operations on Android.
 *
 * @param screenPolygon polygon points in screen coordinates.
 * @param screenWidth screen width in pixels.
 * @param screenHeight screen height in pixels.
 * @return projected area, clipped visible area, and outside flag.
 */
internal actual fun calculateScreenPolygonCoverage(
    screenPolygon: List<ImagePoint>,
    screenWidth: Int,
    screenHeight: Int,
): ZoneScreenCoverageMetrics {
    val screenArea = screenWidth.toFloat() * screenHeight.toFloat()

    if (screenPolygon.size < 3)
        return ZoneScreenCoverageMetrics(0f, 0f, false, screenArea)
    val coordinates = Array(screenPolygon.size + 1) { index ->
        val point = if (index == screenPolygon.size) screenPolygon.first() else screenPolygon[index]
        Coordinate(point.x.toDouble(), point.y.toDouble())
    }
    val polygon = zonePolygonGeometryFactory.createPolygon(coordinates)
    val normalizedPolygon = polygon.buffer(0.0)

    val screenGeometry = zonePolygonGeometryFactory.toGeometry(
        Envelope(0.0, screenWidth.toDouble(), 0.0, screenHeight.toDouble())
    )

    return ZoneScreenCoverageMetrics(
        projectedArea = normalizedPolygon.area.toFloat(),
        visibleArea = normalizedPolygon.intersection(screenGeometry).area.toFloat(),
        isFullyInside = normalizedPolygon.coveredBy(screenGeometry),
        screenArea = screenArea
    )
}

/**
 * Calculates how much of the first polygon is overlapped by the second polygon in screen space.
 *
 * The ratio is normalized by the visible area of the first polygon, which matches the
 * "how much of this zone is covered by another one" interpretation used by frame assignment.
 *
 * @param firstScreenPolygon projected polygon of the candidate zone.
 * @param secondScreenPolygon projected polygon of another candidate zone.
 * @return overlap ratio in `[0, 1]` relative to the first polygon or `0` when geometry is invalid.
 */
internal fun calculateScreenPolygonOcclusionRatio(
    firstScreenPolygon: List<ImagePoint>,
    secondScreenPolygon: List<ImagePoint>,
): Float {
    val firstGeometry = createScreenPolygonGeometry(firstScreenPolygon) ?: return 0f
    val secondGeometry = createScreenPolygonGeometry(secondScreenPolygon) ?: return 0f
    val firstArea = firstGeometry.area.toFloat()
    if (firstArea <= MIN_SCREEN_POLYGON_AREA) {
        return 0f
    }
    val overlapArea = firstGeometry.intersection(secondGeometry).area.toFloat()
    return (overlapArea / firstArea).coerceIn(0f, 1f)
}

/**
 * Builds normalized JTS geometry for one screen-space polygon.
 *
 * @param screenPolygon polygon points in screen pixels.
 * @return normalized polygon geometry or null when input is degenerate.
 */
private fun createScreenPolygonGeometry(
    screenPolygon: List<ImagePoint>,
): Geometry? {
    if (screenPolygon.size < 3) {
        return null
    }
    val coordinates = Array(screenPolygon.size + 1) { index ->
        val point = if (index == screenPolygon.size) screenPolygon.first() else screenPolygon[index]
        Coordinate(point.x.toDouble(), point.y.toDouble())
    }
    val polygon = zonePolygonGeometryFactory.createPolygon(coordinates)
    val normalizedPolygon = polygon.buffer(0.0)
    return if (normalizedPolygon.isEmpty) null else normalizedPolygon
}

private const val MIN_SCREEN_POLYGON_AREA = 1e-6f
