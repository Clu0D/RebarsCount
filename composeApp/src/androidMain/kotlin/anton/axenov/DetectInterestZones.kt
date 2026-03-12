package anton.axenov

import android.graphics.Bitmap
import android.media.Image
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.exceptions.NotYetAvailableException
import dev.romainguy.kotlin.math.Float3
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.node.CubeNode
import kotlin.math.roundToInt
import kotlin.random.Random
import androidx.core.graphics.createBitmap

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

        val boxWidth = (width * random.nextFloat(0.15f, 0.35f)).roundToInt().coerceAtLeast(1)
        val boxHeight = (height * random.nextFloat(0.15f, 0.35f)).roundToInt().coerceAtLeast(1)
        val left = random.nextInt(0, (width - boxWidth).coerceAtLeast(1))
        val top = random.nextInt(0, (height - boxHeight).coerceAtLeast(1))
        val right = (left + boxWidth).coerceAtMost(width - 1)
        val bottom = (top + boxHeight).coerceAtMost(height - 1)
        return listOf(
            DetectedInterestZone(
                boundingBox = BoundingBox(left = left, top = top, right = right, bottom = bottom),
            ),
        )
    }
}

/**
 * Represents one detected interest zone.
 *
 * @param boundingBox zone location in screenshot pixel coordinates.
 */
data class DetectedInterestZone(
    val boundingBox: BoundingBox,
)

/**
 * Represents a pixel-space bounding box.
 *
 * @param left left X pixel coordinate.
 * @param top top Y pixel coordinate.
 * @param right right X pixel coordinate.
 * @param bottom bottom Y pixel coordinate.
 */
data class BoundingBox(
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

/**
 * Snapshot of one frame used for asynchronous detection and world placement.
 *
 * @param screenshot screenshot bitmap captured from this frame.
 * @param frameTimestamp frame timestamp in nanoseconds.
 * @param imageWidth camera image width in pixels.
 * @param imageHeight camera image height in pixels.
 * @param focalLengthX camera focal length X in pixels.
 * @param focalLengthY camera focal length Y in pixels.
 * @param principalPointX camera principal point X in pixels.
 * @param principalPointY camera principal point Y in pixels.
 * @param cameraPose camera pose for this exact frame.
 * @param depthSnapshot depth values sampled from this exact frame.
 */
data class DetectionFrameSnapshot(
    val screenshot: Bitmap,
    val frameTimestamp: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val focalLengthX: Float,
    val focalLengthY: Float,
    val principalPointX: Float,
    val principalPointY: Float,
    val cameraPose: Pose,
    val depthSnapshot: DepthSnapshot,
)

/**
 * Depth map sampled from one frame.
 *
 * Values are raw ARCore `DEPTH16` values.
 *
 * @param width depth image width.
 * @param height depth image height.
 * @param values depth values in row-major order.
 */
data class DepthSnapshot(
    val width: Int,
    val height: Int,
    val values: ShortArray,
)

/**
 * Result of snapshot capture attempt.
 *
 * @param snapshot captured frame snapshot or null when capture failed.
 * @param details capture diagnostic details.
 */
data class DetectionFrameSnapshotCaptureResult(
    val snapshot: DetectionFrameSnapshot?,
    val details: String,
)

/**
 * Placement strategy used to turn a 2D bounding box into a world anchor.
 */
enum class PlacementStrategy {
    DEPTH_SNAPSHOT,
    PLANE_HIT,
    FEATURE_POINT_HIT,
    FAILED,
}

/**
 * Result of world placement attempt for one detected bounding box.
 *
 * @param anchorNode placed anchor node or null when placement failed.
 * @param strategy strategy used for placement.
 * @param details detailed diagnostic information.
 */
data class BoundingBoxPlacementResult(
    val anchorNode: AnchorNode?,
    val strategy: PlacementStrategy,
    val details: String,
)

/**
 * Captures the camera screenshot + depth + camera intrinsics from one ARCore frame.
 *
 * The returned snapshot is immutable and safe to process asynchronously.
 *
 * @param frame ARCore frame to snapshot.
 * @return capture attempt result with snapshot and diagnostics.
 */
fun captureDetectionFrameSnapshot(frame: Frame): DetectionFrameSnapshotCaptureResult {
    val cameraImage = try {
        frame.acquireCameraImage()
    } catch (_: NotYetAvailableException) {
        return DetectionFrameSnapshotCaptureResult(
            snapshot = null,
            details = "Camera image is not yet available",
        )
    } catch (_: IllegalStateException) {
        return DetectionFrameSnapshotCaptureResult(
            snapshot = null,
            details = "Camera image is in invalid state",
        )
    } catch (_: Exception) {
        return DetectionFrameSnapshotCaptureResult(
            snapshot = null,
            details = "Camera image acquisition failed",
        )
    }

    val depthImage = try {
        frame.acquireDepthImage16Bits()
    } catch (_: NotYetAvailableException) {
        cameraImage.close()
        return DetectionFrameSnapshotCaptureResult(
            snapshot = null,
            details = "Depth image is not yet available",
        )
    } catch (_: IllegalStateException) {
        cameraImage.close()
        return DetectionFrameSnapshotCaptureResult(
            snapshot = null,
            details = "Depth image is in invalid state",
        )
    } catch (_: Exception) {
        cameraImage.close()
        return DetectionFrameSnapshotCaptureResult(
            snapshot = null,
            details = "Depth image acquisition failed",
        )
    }

    return try {
        val screenshot = cameraImageToBitmap(cameraImage)
        val depthSnapshot = depthImageToSnapshot(depthImage)
        val intrinsics = frame.camera.imageIntrinsics
        val focalLength = FloatArray(2)
        val principalPoint = FloatArray(2)
        intrinsics.getFocalLength(focalLength, 0)
        intrinsics.getPrincipalPoint(principalPoint, 0)

        DetectionFrameSnapshotCaptureResult(
            snapshot = DetectionFrameSnapshot(
                screenshot = screenshot,
                frameTimestamp = frame.timestamp,
                imageWidth = cameraImage.width,
                imageHeight = cameraImage.height,
                focalLengthX = focalLength[0],
                focalLengthY = focalLength[1],
                principalPointX = principalPoint[0],
                principalPointY = principalPoint[1],
                cameraPose = frame.camera.pose,
                depthSnapshot = depthSnapshot,
            ),
            details =
                "Captured frame snapshot ts=${frame.timestamp}, " +
                    "image=${cameraImage.width}x${cameraImage.height}, " +
                    "depth=${depthSnapshot.width}x${depthSnapshot.height}, " +
                    "fx=${focalLength[0]}, fy=${focalLength[1]}",
        )
    } finally {
        depthImage.close()
        cameraImage.close()
    }
}

/**
 * Places a detected bounding box in world space using depth from the same captured frame snapshot.
 *
 * This method projects the center of the 2D box to a 3D world point using:
 * 1. depth sample at the box center from captured depth map
 * 2. captured camera intrinsics
 * 3. captured camera pose
 *
 * @param sceneView active SceneView AR view.
 * @param snapshot immutable snapshot captured from a specific frame.
 * @param zone detected zone that belongs to the same snapshot screenshot.
 * Fallback order:
 * 1. Depth from captured frame snapshot (frame-consistent).
 * 2. Plane hit test on current frame (not frame-consistent fallback).
 * 3. Feature/depth point hit test on current frame (not frame-consistent fallback).
 *
 * @return placement result with strategy and diagnostics.
 */
fun placeBoundingBoxInWorld(
    sceneView: ARSceneView,
    snapshot: DetectionFrameSnapshot,
    zone: DetectedInterestZone,
): BoundingBoxPlacementResult {
    val depthProjectionAttempt = projectBoundingBoxCenterToWorld(snapshot, zone.boundingBox)
    if (depthProjectionAttempt.worldPoint != null) {
        val session = sceneView.session
        if (session == null) {
            return BoundingBoxPlacementResult(
                anchorNode = null,
                strategy = PlacementStrategy.FAILED,
                details = "Depth projection succeeded, but AR session is null",
            )
        }
        val anchor = session.createAnchor(
            Pose.makeTranslation(
                depthProjectionAttempt.worldPoint[0],
                depthProjectionAttempt.worldPoint[1],
                depthProjectionAttempt.worldPoint[2],
            ),
        )
        val anchorNode = createMarkerAnchorNode(sceneView, anchor)
        return BoundingBoxPlacementResult(
            anchorNode = anchorNode,
            strategy = PlacementStrategy.DEPTH_SNAPSHOT,
            details = "Depth snapshot placement successful. ${depthProjectionAttempt.details}",
        )
    }

    val hitTestCoordinates = mapBoundingBoxCenterToSceneView(snapshot, zone.boundingBox, sceneView)
    if (hitTestCoordinates == null) {
        return BoundingBoxPlacementResult(
            anchorNode = null,
            strategy = PlacementStrategy.FAILED,
            details = "Depth failed. ${depthProjectionAttempt.details}. Current scene size is not ready for fallback hit test.",
        )
    }

    val planeHit = sceneView.hitTestAR(
        xPx = hitTestCoordinates.first,
        yPx = hitTestCoordinates.second,
        planeTypes = setOf(
            Plane.Type.HORIZONTAL_UPWARD_FACING,
            Plane.Type.HORIZONTAL_DOWNWARD_FACING,
            Plane.Type.VERTICAL,
        ),
    )
    if (planeHit != null) {
        val anchorNode = createMarkerAnchorNode(sceneView, planeHit.createAnchor())
        return BoundingBoxPlacementResult(
            anchorNode = anchorNode,
            strategy = PlacementStrategy.PLANE_HIT,
            details =
                "Depth failed. ${depthProjectionAttempt.details}. " +
                    "Fallback plane hit succeeded at view(${hitTestCoordinates.first.toInt()}, ${hitTestCoordinates.second.toInt()}). " +
                    "Note: fallback hit uses current frame, not captured frame ${snapshot.frameTimestamp}.",
        )
    }

    val featurePointHit = sceneView.hitTestAR(
        xPx = hitTestCoordinates.first,
        yPx = hitTestCoordinates.second,
        point = true,
        depthPoint = false,
    )
    if (featurePointHit != null) {
        val anchorNode = createMarkerAnchorNode(sceneView, featurePointHit.createAnchor())
        return BoundingBoxPlacementResult(
            anchorNode = anchorNode,
            strategy = PlacementStrategy.FEATURE_POINT_HIT,
            details =
                "Depth and plane fallback failed. " +
                    "Feature/depth-point hit succeeded at view(${hitTestCoordinates.first.toInt()}, ${hitTestCoordinates.second.toInt()}). " +
                    "Note: fallback hit uses current frame, not captured frame ${snapshot.frameTimestamp}.",
        )
    }

    return BoundingBoxPlacementResult(
        anchorNode = null,
        strategy = PlacementStrategy.FAILED,
        details =
            "Depth failed. ${depthProjectionAttempt.details}. " +
                "Plane hit and feature/depth-point fallback also failed at " +
                "view(${hitTestCoordinates.first.toInt()}, ${hitTestCoordinates.second.toInt()}).",
    )
}

/**
 * Converts one camera image to grayscale bitmap using Y plane values.
 *
 * @param image camera image in `YUV_420_888`.
 * @return ARGB bitmap representing grayscale luminance.
 */
private fun cameraImageToBitmap(image: Image): Bitmap {
    val width = image.width
    val height = image.height
    val plane = image.planes[0]
    val buffer = plane.buffer
    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    val pixels = IntArray(width * height)

    for (y in 0 until height) {
        val rowStart = y * rowStride
        for (x in 0 until width) {
            val luma = buffer.get(rowStart + x * pixelStride).toInt() and 0xFF
            val color = (0xFF shl 24) or (luma shl 16) or (luma shl 8) or luma
            pixels[y * width + x] = color
        }
    }

    return createBitmap(width, height).apply {
        setPixels(pixels, 0, width, 0, 0, width, height)
    }
}

/**
 * Copies ARCore `DEPTH16` image to immutable depth snapshot.
 *
 * @param image depth image returned by `acquireDepthImage16Bits`.
 * @return copied immutable depth snapshot.
 */
private fun depthImageToSnapshot(image: Image): DepthSnapshot {
    val width = image.width
    val height = image.height
    val plane = image.planes[0]
    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    val buffer = plane.buffer.duplicate()
    val values = ShortArray(width * height)

    for (y in 0 until height) {
        val rowStart = y * rowStride
        for (x in 0 until width) {
            val pixelStart = rowStart + x * pixelStride
            val lo = buffer.get(pixelStart).toInt() and 0xFF
            val hi = buffer.get(pixelStart + 1).toInt() and 0xFF
            values[y * width + x] = ((hi shl 8) or lo).toShort()
        }
    }

    return DepthSnapshot(
        width = width,
        height = height,
        values = values,
    )
}

/**
 * Projects a bounding-box center from 2D screenshot space to world coordinates.
 *
 * @param snapshot frame snapshot that owns screenshot and depth data.
 * @param boundingBox bounding box in snapshot screenshot pixel coordinates.
 * @return projection attempt result with optional world point and diagnostics.
 */
private fun projectBoundingBoxCenterToWorld(
    snapshot: DetectionFrameSnapshot,
    boundingBox: BoundingBox,
): DepthProjectionAttempt {
    val imageX = boundingBox.centerX().coerceIn(0, snapshot.imageWidth - 1)
    val imageY = boundingBox.centerY().coerceIn(0, snapshot.imageHeight - 1)
    val depthSample = sampleDepthMeters(snapshot, imageX, imageY)
    val depthMeters = depthSample.depthMeters
        ?: return DepthProjectionAttempt(
            worldPoint = null,
            details = "Depth sample is unavailable at image($imageX, $imageY). ${depthSample.details}",
        )

    // ARCore camera space convention uses -Z forward and +Y up.
    val xCamera = (imageX - snapshot.principalPointX) / snapshot.focalLengthX * depthMeters
    val yCamera = -(imageY - snapshot.principalPointY) / snapshot.focalLengthY * depthMeters
    val zCamera = -depthMeters

    return DepthProjectionAttempt(
        worldPoint = snapshot.cameraPose.transformPoint(floatArrayOf(xCamera, yCamera, zCamera)),
        details =
            "Depth sample: ${depthSample.details}. " +
                "Camera point=($xCamera, $yCamera, $zCamera).",
    )
}

/**
 * Samples depth in meters from the snapshot depth map at image-space coordinate.
 *
 * @param snapshot frame snapshot with depth map.
 * @param imageX X coordinate in screenshot/image pixels.
 * @param imageY Y coordinate in screenshot/image pixels.
 * @return depth sampling attempt with optional depth and diagnostics.
 */
private fun sampleDepthMeters(
    snapshot: DetectionFrameSnapshot,
    imageX: Int,
    imageY: Int,
): DepthSampleAttempt {
    if (snapshot.depthSnapshot.width <= 0 || snapshot.depthSnapshot.height <= 0) {
        return DepthSampleAttempt(
            depthMeters = null,
            details = "Depth map has invalid size ${snapshot.depthSnapshot.width}x${snapshot.depthSnapshot.height}",
        )
    }

    val centerDepthX = ((imageX.toFloat() / snapshot.imageWidth) * snapshot.depthSnapshot.width)
        .toInt()
        .coerceIn(0, snapshot.depthSnapshot.width - 1)
    val centerDepthY = ((imageY.toFloat() / snapshot.imageHeight) * snapshot.depthSnapshot.height)
        .toInt()
        .coerceIn(0, snapshot.depthSnapshot.height - 1)

    var bestDepthMillimeters = 0
    var bestConfidence = -1
    var bestX = centerDepthX
    var bestY = centerDepthY
    var validDepthCount = 0
    var positiveConfidenceCount = 0

    for (dy in -DEPTH_SAMPLE_RADIUS_PX..DEPTH_SAMPLE_RADIUS_PX) {
        for (dx in -DEPTH_SAMPLE_RADIUS_PX..DEPTH_SAMPLE_RADIUS_PX) {
            val sampleX = (centerDepthX + dx).coerceIn(0, snapshot.depthSnapshot.width - 1)
            val sampleY = (centerDepthY + dy).coerceIn(0, snapshot.depthSnapshot.height - 1)
            val rawDepth =
                snapshot.depthSnapshot.values[sampleY * snapshot.depthSnapshot.width + sampleX].toInt() and 0xFFFF
            val depthMillimeters = rawDepth and DEPTH16_DEPTH_MASK
            val confidence = (rawDepth and DEPTH16_CONFIDENCE_MASK) ushr DEPTH16_CONFIDENCE_SHIFT
            if (depthMillimeters > 0) {
                validDepthCount++
                if (confidence > 0) {
                    positiveConfidenceCount++
                }
                if (confidence > bestConfidence || (confidence == bestConfidence && depthMillimeters > bestDepthMillimeters)) {
                    bestConfidence = confidence
                    bestDepthMillimeters = depthMillimeters
                    bestX = sampleX
                    bestY = sampleY
                }
            }
        }
    }

    if (bestDepthMillimeters <= 0) {
        return DepthSampleAttempt(
            depthMeters = null,
            details =
                "No valid depth in neighborhood. " +
                    "image($imageX,$imageY) -> centerDepth($centerDepthX,$centerDepthY), " +
                    "windowRadius=$DEPTH_SAMPLE_RADIUS_PX, validDepthCount=$validDepthCount",
        )
    }

    return DepthSampleAttempt(
        depthMeters = bestDepthMillimeters / MILLIMETERS_IN_METER,
        details =
            "image($imageX,$imageY) -> bestDepth($bestX,$bestY), " +
                "windowRadius=$DEPTH_SAMPLE_RADIUS_PX, " +
                "depthMm=$bestDepthMillimeters, confidence=$bestConfidence, " +
                "validDepthCount=$validDepthCount, positiveConfidenceCount=$positiveConfidenceCount",
    )
}

/**
 * Maps screenshot-space bounding box center to SceneView pixel coordinates.
 *
 * @param snapshot frame snapshot that owns screenshot dimensions.
 * @param boundingBox bounding box in screenshot coordinates.
 * @param sceneView target SceneView.
 * @return pair of `(xPx, yPx)` in SceneView coordinates or null if view is not ready.
 */
private fun mapBoundingBoxCenterToSceneView(
    snapshot: DetectionFrameSnapshot,
    boundingBox: BoundingBox,
    sceneView: ARSceneView,
): Pair<Float, Float>? {
    if (sceneView.width <= 0 || sceneView.height <= 0) {
        return null
    }
    if (snapshot.imageWidth <= 0 || snapshot.imageHeight <= 0) {
        return null
    }
    val normalizedX = boundingBox.centerX().toFloat() / snapshot.imageWidth.toFloat()
    val normalizedY = boundingBox.centerY().toFloat() / snapshot.imageHeight.toFloat()
    return Pair(
        first = normalizedX.coerceIn(0f, 1f) * sceneView.width.toFloat(),
        second = normalizedY.coerceIn(0f, 1f) * sceneView.height.toFloat(),
    )
}

/**
 * Creates a marker anchor node and adds it to the scene.
 *
 * @param sceneView active SceneView.
 * @param anchor anchor to attach marker node to.
 * @return created anchor node.
 */
private fun createMarkerAnchorNode(
    sceneView: ARSceneView,
    anchor: com.google.ar.core.Anchor,
): AnchorNode {
    val anchorNode = AnchorNode(sceneView.engine, anchor)
    val markerNode = CubeNode(
        engine = sceneView.engine,
        size = Float3(0.12f, 0.12f, 0.12f),
    )
    anchorNode.addChildNode(markerNode)
    sceneView.addChildNode(anchorNode)
    return anchorNode
}

/**
 * Result of 2D-to-3D depth projection for a bounding-box center.
 *
 * @param worldPoint projected world point or null when projection failed.
 * @param details detailed projection diagnostics.
 */
private data class DepthProjectionAttempt(
    val worldPoint: FloatArray?,
    val details: String,
)

/**
 * Result of one depth sampling attempt.
 *
 * @param depthMeters sampled depth in meters or null when unavailable.
 * @param details detailed depth sampling diagnostics.
 */
private data class DepthSampleAttempt(
    val depthMeters: Float?,
    val details: String,
)

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

private const val DEPTH16_DEPTH_MASK = 0x1FFF
private const val DEPTH16_CONFIDENCE_MASK = 0xE000
private const val DEPTH16_CONFIDENCE_SHIFT = 13
private const val MILLIMETERS_IN_METER = 1000f
private const val DEPTH_SAMPLE_RADIUS_PX = 4
