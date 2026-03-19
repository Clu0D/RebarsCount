package anton.axenov

import android.graphics.Bitmap
import kotlin.math.roundToInt
import kotlin.random.Random

/**
 * Detects zones of interest on a screenshot.
 *
 * This is a dummy implementation that returns exactly one random bounding box.
 *
 * @param random random source used for deterministic tests when needed.
 */
class DetectInterestZones(
    private val random: Random = Random.Default,
) {
    /**
     * Detects zones of interest on a screenshot.
     *
     * @param screenshot screenshot captured from camera frame.
     * @return list with one random zone bounding box.
     */
    fun detectZones(screenshot: Bitmap): List<DetectedInterestZone> {
        val width = screenshot.width
        val height = screenshot.height
        if (width <= 1 || height <= 1) {
            return emptyList()
        }

        val boxWidth = (width * random.nextFloat(DETECTION_BOX_MIN_RATIO, DETECTION_BOX_MAX_RATIO))
            .roundToInt()
            .coerceAtLeast(1)
        val boxHeight = (height * random.nextFloat(DETECTION_BOX_MIN_RATIO, DETECTION_BOX_MAX_RATIO))
            .roundToInt()
            .coerceAtLeast(1)
        val left = random.nextInt(0, (width - boxWidth).coerceAtLeast(1))
        val top = random.nextInt(0, (height - boxHeight).coerceAtLeast(1))
        val right = (left + boxWidth).coerceAtMost(width - 1)
        val bottom = (top + boxHeight).coerceAtMost(height - 1)
        return listOf(
            DetectedInterestZone(
                boundingBox = translateCoordinates(left, top, right, bottom, width, height),
            ),
        )
    }

    /**
     * Translates detector-space coordinates to image-space coordinates used by placement math.
     *
     * Detector output is treated as using the same X axis but opposite Y axis direction
     * (origin at bottom-left). Placement and depth projection code expect top-left image origin.
     * Inputs are clamped to image bounds and reordered if detector produces inverted edges.
     *
     * @param left detector-space left X coordinate.
     * @param top detector-space top Y coordinate.
     * @param right detector-space right X coordinate.
     * @param bottom detector-space bottom Y coordinate.
     * @param width source image width in pixels.
     * @param height source image height in pixels.
     * @return translated bounding box in image-space coordinates.
     */
    fun translateCoordinates(left: Int, top: Int, right: Int, bottom: Int, width: Int, height: Int): BoundingBox {
        if (width <= 0 || height <= 0) {
            return BoundingBox(left = 0, top = 0, right = 0, bottom = 0)
        }

        val maxX = width - 1
        val maxY = height - 1
        val clampedLeft = left.coerceIn(0, maxX)
        val clampedRight = right.coerceIn(0, maxX)
        val clampedTop = top.coerceIn(0, maxY)
        val clampedBottom = bottom.coerceIn(0, maxY)

        val normalizedLeft = minOf(clampedLeft, clampedRight)
        val normalizedRight = maxOf(clampedLeft, clampedRight)
        val normalizedTop = minOf(clampedTop, clampedBottom)
        val normalizedBottom = maxOf(clampedTop, clampedBottom)

        val newLeft = normalizedLeft
        val newTop = maxY - normalizedBottom
        val newRight = normalizedRight
        val newBottom = maxY - normalizedTop
        return BoundingBox(newLeft, newTop, newRight, newBottom)
    }
}

/**
 * Returns a random float inside inclusive range.
 *
 * @param start inclusive start value.
 * @param end inclusive end value.
 * @return random float between start and end.
 */
private fun Random.nextFloat(start: Float, end: Float): Float {
    return start + nextFloat() * (end - start)
}

private const val DETECTION_BOX_MIN_RATIO = 0.15f
private const val DETECTION_BOX_MAX_RATIO = 0.35f
