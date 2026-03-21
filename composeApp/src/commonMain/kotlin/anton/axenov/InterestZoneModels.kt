package anton.axenov

/**
 * Represents one detected interest zone.
 *
 * @param screenBoundingBox zone location in screenshot pixel coordinates.
 */
data class DetectedInterestZone(
    val screenBoundingBox: ScreenBoundingBox,
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
