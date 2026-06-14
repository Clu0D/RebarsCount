package anton.axenov

import kotlin.math.abs

/**
 * Calculates polygon coverage on native targets.
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
    TODO("Not yet implemented")
}
