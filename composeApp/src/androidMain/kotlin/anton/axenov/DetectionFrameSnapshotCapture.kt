package anton.axenov

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream

/**
 * Captures the camera screenshot + optional depth + camera intrinsics from one ARCore frame.
 *
 * The returned snapshot is immutable and safe to process asynchronously.
 *
 * @param frame ARCore frame to snapshot.
 * @param distortionCoefficients lens distortion coefficients.
 * @return capture attempt result with snapshot and diagnostics.
 */
fun captureDetectionFrameSnapshot(
    frame: Frame,
    distortionCoefficients: List<Float> = emptyList(),
): DetectionFrameSnapshotCaptureResult {
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
        val encodedScreenshot = cameraImageToJpegByteArray(cameraImage)
        val screenshot = BitmapFactory.decodeByteArray(
            encodedScreenshot,
            0,
            encodedScreenshot.size,
        )
        val depthSnapshot = depthImage?.let(::depthImageToSnapshot)
        val intrinsics = frame.camera.imageIntrinsics
        val focalLength = FloatArray(2)
        val principalPoint = FloatArray(2)
        intrinsics.getFocalLength(focalLength, 0)
        intrinsics.getPrincipalPoint(principalPoint, 0)

        DetectionFrameSnapshotCaptureResult(
            snapshot = DetectionFrameSnapshot(
                screenshot = screenshot,
                screenshotJpegBytes = encodedScreenshot,
                frameTimestamp = frame.timestamp,
                imageWidth = cameraImage.width,
                imageHeight = cameraImage.height,
                focalLengthX = focalLength[0],
                focalLengthY = focalLength[1],
                principalPointX = principalPoint[0],
                principalPointY = principalPoint[1],
                distortionCoefficients = distortionCoefficients,
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
 * Converts one camera image from `YUV_420_888` into JPEG bytes.
 *
 * @param image camera image in `YUV_420_888`.
 * @return JPEG-encoded image bytes.
 */
private fun cameraImageToJpegByteArray(image: Image): ByteArray {
    // Get the three planes from the YUV_420_888 image
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer

    // Combine all planes into a single byte array
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    // Convert to JPEG then to Bitmap
    val outputStream = ByteArrayOutputStream()
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, outputStream)
    return outputStream.toByteArray()
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
