package anton.axenov

import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Envelope
import org.locationtech.jts.geom.GeometryFactory


private val gf = GeometryFactory()

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
    val polygon = gf.createPolygon(coordinates)
    val normalizedPolygon = polygon.buffer(0.0)

    val screenGeometry = gf.toGeometry(
        Envelope(0.0, screenWidth.toDouble(), 0.0, screenHeight.toDouble())
    )

    return ZoneScreenCoverageMetrics(
        projectedArea = normalizedPolygon.area.toFloat(),
        visibleArea = normalizedPolygon.intersection(screenGeometry).area.toFloat(),
        isFullyInside = normalizedPolygon.coveredBy(screenGeometry),
        screenArea = screenArea
    )
}
