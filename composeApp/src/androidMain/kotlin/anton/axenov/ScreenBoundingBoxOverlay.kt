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
 * One temporary screen-space overlay bounding box drawn over the AR view.
 *
 * @param screenBoundingBox polygon bounds in source image pixels.
 * @param sourceImageWidth source image width in pixels.
 * @param sourceImageHeight source image height in pixels.
 * @param expiresAtElapsedMs elapsed realtime timestamp when overlay should disappear.
 */
data class ScreenBoundingBoxOverlayEntry(
    val screenBoundingBox: ScreenBoundingBox,
    val sourceImageWidth: Int,
    val sourceImageHeight: Int,
    val expiresAtElapsedMs: Long,
)

/**
 * Keeps screen-space detection overlays short-lived and separate from AR placement code.
 *
 * @param overlayLifetimeMs overlay visibility duration in milliseconds.
 */
class ScreenPolygonOverlayStore(
    private val overlayLifetimeMs: Long = SCREEN_POLYGON_LIFETIME_MS,
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
                screenBoundingBox = zone.screenBoundingBox,
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
 * Draws temporary detection polygons in screen pixel space above the AR view.
 *
 * @param overlays active overlays to draw.
 * @param modifier overlay modifier.
 */
@Composable
fun ScreenPolygonOverlay(
    overlays: List<ScreenBoundingBoxOverlayEntry>,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        overlays.forEach { overlay ->
            if (overlay.sourceImageWidth <= 0 || overlay.sourceImageHeight <= 0) {
                return@forEach
            }
            val mappedCorners = mapImagePointsToViewPoints(
                imagePoints = listOf(
                    ImagePoint(overlay.screenBoundingBox.left, overlay.screenBoundingBox.top),
                    ImagePoint(overlay.screenBoundingBox.right, overlay.screenBoundingBox.top),
                    ImagePoint(overlay.screenBoundingBox.right, overlay.screenBoundingBox.bottom),
                    ImagePoint(overlay.screenBoundingBox.left, overlay.screenBoundingBox.bottom),
                ),
                imageWidth = overlay.sourceImageWidth,
                imageHeight = overlay.sourceImageHeight,
                viewWidth = size.width.toInt(),
                viewHeight = size.height.toInt(),
            )
            if (mappedCorners.size < 4) {
                return@forEach
            }
            val left = mappedCorners.minOf { it.xPx }
            val top = mappedCorners.minOf { it.yPx }
            val right = mappedCorners.maxOf { it.xPx }
            val bottom = mappedCorners.maxOf { it.yPx }
            val strokeWidth = (size.minDimension * SCREEN_POLYGON_STROKE_RATIO)
                .coerceIn(SCREEN_POLYGON_MIN_STROKE_PX, SCREEN_POLYGON_MAX_STROKE_PX)

            drawRect(
                color = SCREEN_POLYGON_COLOR,
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

private const val SCREEN_POLYGON_LIFETIME_MS = 5_000L
private const val SCREEN_POLYGON_STROKE_RATIO = 0.004f
private const val SCREEN_POLYGON_MIN_STROKE_PX = 2f
private const val SCREEN_POLYGON_MAX_STROKE_PX = 8f
private val SCREEN_POLYGON_COLOR = Color(0xFFFF6B35)
