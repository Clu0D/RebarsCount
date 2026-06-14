package anton.axenov

/**
 * Calculates polygon coverage using JTS geometry operations on JVM.
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