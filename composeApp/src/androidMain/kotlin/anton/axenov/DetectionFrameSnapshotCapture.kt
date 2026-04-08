package anton.axenov

import android.graphics.Bitmap
import android.media.Image
import androidx.core.graphics.createBitmap
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException

/**
 * Captures the camera screenshot + optional depth + camera intrinsics from one ARCore frame.
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

    val depthCapture: Pair<Image?, String> = try {
        Pair(frame.acquireDepthImage16Bits(), "available")
    } catch (_: NotYetAvailableException) {
        Pair(null, "not yet available")
    } catch (_: IllegalStateException) {
        Pair(null, "invalid state")
    } catch (_: Exception) {
        Pair(null, "acquisition failed")
    }
    val depthImage = depthCapture.first
    val depthStatus = depthCapture.second

    return try {
        val screenshot = cameraImageToBitmap(cameraImage)
        val depthSnapshot = depthImage?.let(::depthImageToSnapshot)
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
                    if (depthSnapshot != null) {
                        "depth=${depthSnapshot.width}x${depthSnapshot.height}, "
                    } else {
                        "depth=unavailable($depthStatus), "
                    } +
                    "fx=${focalLength[0]}, fy=${focalLength[1]}",
        )
    } finally {
        depthImage?.close()
        cameraImage.close()
    }
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
 * Creates immutable bitmap copy suitable for long-lived storage.
 *
 * @return copied immutable bitmap.
 */
fun Bitmap.copyBitmapForStorage(): Bitmap {
    return copy(config ?: Bitmap.Config.ARGB_8888, false)
}