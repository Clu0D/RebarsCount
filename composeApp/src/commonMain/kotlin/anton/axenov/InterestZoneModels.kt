package anton.axenov

/**
 * Represents one detected interest zone.
 *
 * @param screenBoundingBox zone location in screenshot pixel coordinates.
 * @param confidence segmentation confidence in range `[0, 1]`.
 */
data class DetectedInterestZone(
    val screenBoundingBox: ScreenBoundingBox,
    val confidence: Float,
)

/**
 * Represents a pixel-space bounding box.
 *
 * @param left left X pixel coordinate.
 * @param top top Y pixel coordinate.
 * @param right right X pixel coordinate.
 * @param bottom bottom Y pixel coordinate.
 */
data class ScreenBoundingBox(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
) {
    constructor(polygon: List<ImagePoint>) : this(
        left = polygon.minOfOrNull { it.x } ?: 0,
        top = polygon.minOfOrNull { it.y } ?: 0,
        right = polygon.maxOfOrNull { it.x } ?: 0,
        bottom = polygon.maxOfOrNull { it.y } ?: 0,
    )

    /**
     * Returns horizontal center in pixels.
     *
     * @return center X pixel coordinate.
     */
    fun centerX(): Int = (left + right) / 2

    /**
     * Returns vertical center in pixels.
     *
     * @return center Y pixel coordinate.
     */
    fun centerY(): Int = (top + bottom) / 2
}
