package anton.axenov

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * Formats translation/orientation debug text with image and view dimensions.
 *
 * @param orientationName orientation text to render.
 * @param imageWidth captured camera image width in pixels.
 * @param imageHeight captured camera image height in pixels.
 * @param viewWidth AR view width in pixels.
 * @param viewHeight AR view height in pixels.
 * @return formatted overlay line.
 */
fun formatTranslationOverlayText(
    orientationName: String,
    imageWidth: Int,
    imageHeight: Int,
    viewWidth: Int,
    viewHeight: Int,
): String {
    val safeImageWidth = imageWidth.coerceAtLeast(0)
    val safeImageHeight = imageHeight.coerceAtLeast(0)
    val safeViewWidth = viewWidth.coerceAtLeast(0)
    val safeViewHeight = viewHeight.coerceAtLeast(0)
    return "[$orientationName, ${safeImageWidth}x${safeImageHeight} img, ${safeViewWidth}x${safeViewHeight} view]"
}

fun Float.toPrecision(precision: Int) =
    this.toDouble().toPrecision(precision)

fun Double.toPrecision(precision: Int) =
    if (precision < 1) {
        "${this.roundToInt()}"
    } else {
        val p = 10.0.pow(precision)
        val v = (abs(this) * p).roundToInt()
        val i = floor(v / p)
        var f = "${floor(v - (i * p)).toInt()}"
        while (f.length < precision) f = "0$f"
        val s = if (this < 0) "-" else ""
        "$s${i.toInt()}.$f"
    }
