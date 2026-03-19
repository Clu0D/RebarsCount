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
    fun detectZones(
        screenshot: Bitmap
    ): List<DetectedInterestZone> {
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
                boundingBox = translateCoordinates(
                    left = left,
                    top = top,
                    right = right,
                    bottom = bottom,
                ),
            ),
        )
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
