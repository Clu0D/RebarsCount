package anton.axenov

import android.os.SystemClock
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke

/**
 * One temporary screen-space overlay rectangle drawn over the AR view.
 *
 * @param boundingBox rectangle in source image pixels.
 * @param sourceImageWidth source image width in pixels.
 * @param sourceImageHeight source image height in pixels.
 * @param expiresAtElapsedMs elapsed realtime timestamp when overlay should disappear.
 */
data class ScreenBoundingBoxOverlayEntry(
    val boundingBox: BoundingBox,
    val sourceImageWidth: Int,
    val sourceImageHeight: Int,
    val expiresAtElapsedMs: Long,
)

/**
 * Keeps screen-space detection overlays short-lived and separate from AR placement code.
 *
 * @param overlayLifetimeMs overlay visibility duration in milliseconds.
 */
class ScreenBoundingBoxOverlayStore(
    private val overlayLifetimeMs: Long = SCREEN_BOUNDING_BOX_LIFETIME_MS,
) {
    /**
     * Adds newly detected zones as screen overlays and drops already expired ones.
     *
     * @param current currently active overlays.
     * @param snapshot detection frame snapshot.
     * @param zones newly detected zones.
     * @param nowElapsedMs current elapsed realtime timestamp.
     * @return active overlays list with newly appended entries.
     */
    fun addDetectedZones(
        current: List<ScreenBoundingBoxOverlayEntry>,
        snapshot: DetectionFrameSnapshot,
        zones: List<DetectedInterestZone>,
        nowElapsedMs: Long = SystemClock.elapsedRealtime(),
    ): List<ScreenBoundingBoxOverlayEntry> {
        val activeOverlays = pruneExpired(current, nowElapsedMs)
        val newOverlays = zones.map { zone ->
            ScreenBoundingBoxOverlayEntry(
                boundingBox = zone.boundingBox,
                sourceImageWidth = snapshot.imageWidth,
                sourceImageHeight = snapshot.imageHeight,
                expiresAtElapsedMs = nowElapsedMs + overlayLifetimeMs,
            )
        }
        return activeOverlays + newOverlays
    }

    /**
     * Removes overlays whose lifetime already elapsed.
     */
    fun pruneExpired(
        current: List<ScreenBoundingBoxOverlayEntry>,
        nowElapsedMs: Long = SystemClock.elapsedRealtime(),
    ): List<ScreenBoundingBoxOverlayEntry> {
        return current.filter { overlay -> overlay.expiresAtElapsedMs > nowElapsedMs }
    }
}

/**
 * Draws temporary detection rectangles in screen pixel space above the AR view.
 *
 * @param overlays active overlays to draw.
 * @param modifier overlay modifier.
 */
@Composable
fun ScreenBoundingBoxOverlay(
    overlays: List<ScreenBoundingBoxOverlayEntry>,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        overlays.forEach { overlay ->
            if (overlay.sourceImageWidth <= 0 || overlay.sourceImageHeight <= 0) {
                return@forEach
            }

            val scaleX = size.width / overlay.sourceImageWidth.toFloat()
            val scaleY = size.height / overlay.sourceImageHeight.toFloat()
            val left = overlay.boundingBox.left * scaleX
            val top = overlay.boundingBox.top * scaleY
            val right = overlay.boundingBox.right * scaleX
            val bottom = overlay.boundingBox.bottom * scaleY
            val strokeWidth = (size.minDimension * SCREEN_BOUNDING_BOX_STROKE_RATIO)
                .coerceIn(SCREEN_BOUNDING_BOX_MIN_STROKE_PX, SCREEN_BOUNDING_BOX_MAX_STROKE_PX)

            drawRect(
                color = SCREEN_BOUNDING_BOX_COLOR,
                topLeft = Offset(left, top),
                size = Size(
                    width = (right - left).coerceAtLeast(1f),
                    height = (bottom - top).coerceAtLeast(1f),
                ),
                style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
            )
        }
    }
}

private const val SCREEN_BOUNDING_BOX_LIFETIME_MS = 5_000L
private const val SCREEN_BOUNDING_BOX_STROKE_RATIO = 0.004f
private const val SCREEN_BOUNDING_BOX_MIN_STROKE_PX = 2f
private const val SCREEN_BOUNDING_BOX_MAX_STROKE_PX = 8f
private val SCREEN_BOUNDING_BOX_COLOR = Color(0xFFFF6B35)
