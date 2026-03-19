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
     * @param translationVariant detector-to-image translation variant based on current device orientation.
     * @return list with one random zone bounding box.
     */
    fun detectZones(
        screenshot: Bitmap,
        translationVariant: CoordinateTranslationVariant,
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
                    width = width,
                    height = height,
                    translationVariant = translationVariant,
                ),
            ),
        )
    }

    /**
     * Translates detector-space coordinates to image-space coordinates used by placement math.
     *
     * @param left detector-space left X coordinate.
     * @param top detector-space top Y coordinate.
     * @param right detector-space right X coordinate.
     * @param bottom detector-space bottom Y coordinate.
     * @param width source image width in pixels.
     * @param height source image height in pixels.
     * @param translationVariant detector-to-image translation variant.
     * @return translated bounding box in image-space coordinates.
     */
    fun translateCoordinates(
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        width: Int,
        height: Int,
        translationVariant: CoordinateTranslationVariant,
    ): BoundingBox {

        return when (translationVariant) {
            CoordinateTranslationVariant.PORTRAIT -> BoundingBox(
                left = top,
                top = left,
                right = bottom,
                bottom = right,
            )

            CoordinateTranslationVariant.LANDSCAPE -> BoundingBox(
                left = width - 1 - right,
                top = height - 1 - bottom,
                right = width - 1 - left,
                bottom = height - 1 - top,
            )

            CoordinateTranslationVariant.LANDSCAPE_REVERSED -> BoundingBox(
                left = width - 1 - right,
                top = top,
                right = width - 1 - left,
                bottom = bottom,
            )
        }
    }
}

/**
 * Translation strategy by orientation.
 */
enum class CoordinateTranslationVariant {
    PORTRAIT,
    LANDSCAPE,
    LANDSCAPE_REVERSED,
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
