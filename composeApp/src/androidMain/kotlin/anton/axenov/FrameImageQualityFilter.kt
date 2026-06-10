package anton.axenov

import android.os.SystemClock
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * AR-derived camera pose sample used by [FrameImageQualityFilter].
 *
 * @param translationX camera translation X in world meters.
 * @param translationY camera translation Y in world meters.
 * @param translationZ camera translation Z in world meters.
 * @param rotationX quaternion X component.
 * @param rotationY quaternion Y component.
 * @param rotationZ quaternion Z component.
 * @param rotationW quaternion W component.
 */
data class FrameCameraPoseSample(
    val translationX: Float,
    val translationY: Float,
    val translationZ: Float,
    val rotationX: Float,
    val rotationY: Float,
    val rotationZ: Float,
    val rotationW: Float,
)

/**
 * Applies frame-quality checks before expensive downstream processing starts.
 *
 * Camera stability is derived from AR pose deltas. Exposure and sharpness are
 * measured with OpenCV on the encoded frame image.
 *
 * @param minAcceptedIntervalMs maximum time without one accepted frame.
 * @param maxUnderexposedRatio maximum acceptable ratio of very dark pixels.
 * @param maxOverexposedRatio maximum acceptable ratio of very bright pixels.
 * @param minLumaStdDev minimum acceptable luma standard deviation.
 * @param minLaplacianVariance minimum acceptable variance of Laplacian blur metric.
 * @param maxTranslationDeltaMeters maximum acceptable camera translation delta.
 * @param maxRotationDeltaDegrees maximum acceptable camera rotation delta.
 */
class FrameImageQualityFilter(
    private val minAcceptedIntervalMs: Long = DEFAULT_MIN_ACCEPTED_INTERVAL_MS,
    private val maxUnderexposedRatio: Float = DEFAULT_MAX_UNDEREXPOSED_RATIO,
    private val maxOverexposedRatio: Float = DEFAULT_MAX_OVEREXPOSED_RATIO,
    private val minLumaStdDev: Float = DEFAULT_MIN_LUMA_STD_DEV,
    private val minLaplacianVariance: Float = DEFAULT_MIN_LAPLACIAN_VARIANCE,
    private val maxTranslationDeltaMeters: Float = DEFAULT_MAX_TRANSLATION_DELTA_METERS,
    private val maxRotationDeltaDegrees: Float = DEFAULT_MAX_ROTATION_DELTA_DEGREES,
) {
    private var lastAcceptedAtMs: Long? = null
    private var previousPose: FrameCameraPoseSample? = null

    /**
     * Evaluates one frame and decides whether it may continue through the pipeline.
     *
     * @param encodedImageBytes encoded frame image bytes.
     * @param cameraPose current AR-derived camera pose sample.
     * @param nowElapsedMs current monotonic time in milliseconds.
     * @return frame-quality decision with metrics and reject reasons.
     */
    fun evaluate(
        encodedImageBytes: ByteArray,
        cameraPose: FrameCameraPoseSample,
        nowElapsedMs: Long = SystemClock.elapsedRealtime(),
    ): FrameImageQualityDecision {
        val imageMetrics = calculateFrameImageQualityMetrics(encodedImageBytes)
        val motionMetrics = calculateMotionMetrics(previousPose, cameraPose)
        previousPose = cameraPose

        val rejectReasons = buildList {
            if (imageMetrics != null) {
                if (imageMetrics.underexposedRatio > maxUnderexposedRatio ||
                    imageMetrics.overexposedRatio > maxOverexposedRatio
                ) {
                    add("exposure")
                }
                if (imageMetrics.lumaStdDev < minLumaStdDev) {
                    add("contrast")
                }
                if (imageMetrics.laplacianVariance < minLaplacianVariance) {
                    add("sharpness")
                }
            }
            if (motionMetrics.translationDeltaMeters > maxTranslationDeltaMeters ||
                motionMetrics.rotationDeltaDegrees > maxRotationDeltaDegrees
            ) {
                add("camera_motion")
            }
        }

        val passesNormally = rejectReasons.isEmpty()
        val shouldForceAccept = !passesNormally &&
            lastAcceptedAtMs?.let { acceptedAtMs ->
                nowElapsedMs - acceptedAtMs >= minAcceptedIntervalMs
            } ?: true

        val isAccepted = passesNormally || shouldForceAccept
        if (isAccepted) {
            lastAcceptedAtMs = nowElapsedMs
        }

        return FrameImageQualityDecision(
            isAccepted = isAccepted,
            isForcedByTimeout = !passesNormally && shouldForceAccept,
            rejectReasons = rejectReasons,
            metrics = FrameImageQualityMetrics(
                meanLuma = imageMetrics?.meanLuma ?: 0f,
                lumaStdDev = imageMetrics?.lumaStdDev ?: 0f,
                underexposedRatio = imageMetrics?.underexposedRatio ?: 0f,
                overexposedRatio = imageMetrics?.overexposedRatio ?: 0f,
                laplacianVariance = imageMetrics?.laplacianVariance ?: 0f,
                translationDeltaMeters = motionMetrics.translationDeltaMeters,
                rotationDeltaDegrees = motionMetrics.rotationDeltaDegrees,
                imageMetricsAvailable = imageMetrics != null,
            ),
        )
    }

    /**
     * Clears internal history between sessions.
     */
    fun reset() {
        lastAcceptedAtMs = null
        previousPose = null
    }

    /**
     * Calculates camera motion between previous and current AR pose samples.
     *
     * @param previousPose previous evaluated pose or null for the first frame.
     * @param currentPose current pose sample.
     * @return translation and rotation deltas.
     */
    private fun calculateMotionMetrics(
        previousPose: FrameCameraPoseSample?,
        currentPose: FrameCameraPoseSample,
    ): MotionMetrics {
        if (previousPose == null) {
            return MotionMetrics()
        }

        val dx = currentPose.translationX - previousPose.translationX
        val dy = currentPose.translationY - previousPose.translationY
        val dz = currentPose.translationZ - previousPose.translationZ
        val translationDeltaMeters = sqrt(dx * dx + dy * dy + dz * dz)

        val quaternionDot = (
            currentPose.rotationX * previousPose.rotationX +
                currentPose.rotationY * previousPose.rotationY +
                currentPose.rotationZ * previousPose.rotationZ +
                currentPose.rotationW * previousPose.rotationW
            ).coerceIn(-1f, 1f)
        val rotationDeltaRadians = 2.0 * acos(abs(quaternionDot).coerceIn(0f, 1f).toDouble())
        val rotationDeltaDegrees = (rotationDeltaRadians * 180.0 / PI).toFloat()

        return MotionMetrics(
            translationDeltaMeters = translationDeltaMeters,
            rotationDeltaDegrees = rotationDeltaDegrees,
        )
    }
}

/**
 * One frame-quality decision returned by [FrameImageQualityFilter].
 *
 * @param isAccepted true when the frame may continue through processing.
 * @param isForcedByTimeout true when the frame was accepted only by starvation guard.
 * @param rejectReasons symbolic reasons that blocked normal acceptance.
 * @param metrics measured image and motion values.
 */
data class FrameImageQualityDecision(
    val isAccepted: Boolean,
    val isForcedByTimeout: Boolean,
    val rejectReasons: List<String>,
    val metrics: FrameImageQualityMetrics,
) {
    /**
     * Builds a compact debug string for logs and overlays.
     *
     * @return short user-visible description.
     */
    fun toDebugText(): String {
        val reasonsText = if (rejectReasons.isEmpty()) "ok" else rejectReasons.joinToString(",")
        val forcedText = if (isForcedByTimeout) ", forced=yes" else ""
        val imageText = if (metrics.imageMetricsAvailable) {
            "luma=${metrics.meanLuma.roundToInt()}, " +
                "std=${metrics.lumaStdDev.formatToDigits(1)}, " +
                "dark=${(metrics.underexposedRatio * 100f).formatToDigits(1)}%, " +
                "bright=${(metrics.overexposedRatio * 100f).formatToDigits(1)}%, " +
                "lap=${metrics.laplacianVariance.formatToDigits(1)}, "
        } else {
            "imageMetrics=unavailable, "
        }
        return "reasons=$reasonsText, " +
            imageText +
            "move=${metrics.translationDeltaMeters.formatToDigits(3)}m, " +
            "rot=${metrics.rotationDeltaDegrees.formatToDigits(1)}deg" +
            forcedText
    }
}

/**
 * Aggregated frame-quality metrics used by [FrameImageQualityFilter].
 *
 * @param meanLuma average image luma in `0..255`.
 * @param lumaStdDev luma standard deviation.
 * @param underexposedRatio ratio of very dark pixels.
 * @param overexposedRatio ratio of very bright pixels.
 * @param laplacianVariance blur metric based on Laplacian variance.
 * @param translationDeltaMeters camera translation since previous frame.
 * @param rotationDeltaDegrees camera rotation since previous frame.
 * @param imageMetricsAvailable true when image metrics were calculated successfully.
 */
data class FrameImageQualityMetrics(
    val meanLuma: Float,
    val lumaStdDev: Float,
    val underexposedRatio: Float,
    val overexposedRatio: Float,
    val laplacianVariance: Float,
    val translationDeltaMeters: Float,
    val rotationDeltaDegrees: Float,
    val imageMetricsAvailable: Boolean,
)

/**
 * Image-only metrics computed by OpenCV.
 *
 * @param meanLuma average image luma in `0..255`.
 * @param lumaStdDev luma standard deviation.
 * @param underexposedRatio ratio of pixels below the dark threshold.
 * @param overexposedRatio ratio of pixels above the bright threshold.
 * @param laplacianVariance variance of the Laplacian response.
 */
private data class OpenCvFrameImageQualityMetrics(
    val meanLuma: Float,
    val lumaStdDev: Float,
    val underexposedRatio: Float,
    val overexposedRatio: Float,
    val laplacianVariance: Float,
)

/**
 * Calculates image quality metrics with OpenCV on encoded image bytes.
 *
 * @param encodedImageBytes encoded frame image bytes.
 * @return image metrics or null when OpenCV is unavailable or decode failed.
 */
private fun calculateFrameImageQualityMetrics(
    encodedImageBytes: ByteArray,
): OpenCvFrameImageQualityMetrics? {
    if (!isOpenCvReady()) {
        return null
    }

    val encodedMat = Mat(1, encodedImageBytes.size, CvType.CV_8U)
    val grayMat = Mat()
    val laplacianMat = Mat()
    val meanMat = MatOfDouble()
    val stdDevMat = MatOfDouble()

    return try {
        encodedMat.put(0, 0, encodedImageBytes)
        val decodedMat = Imgcodecs.imdecode(encodedMat, Imgcodecs.IMREAD_GRAYSCALE)
        decodedMat.copyTo(grayMat)
        decodedMat.release()
        if (grayMat.empty()) {
            return null
        }

        Core.meanStdDev(grayMat, meanMat, stdDevMat)
        val meanLuma = meanMat.toArray().firstOrNull()?.toFloat() ?: return null
        val lumaStdDev = stdDevMat.toArray().firstOrNull()?.toFloat() ?: return null

        Imgproc.Laplacian(grayMat, laplacianMat, CvType.CV_64F)
        val laplacianMean = MatOfDouble()
        val laplacianStdDev = MatOfDouble()
        try {
            Core.meanStdDev(laplacianMat, laplacianMean, laplacianStdDev)
            val laplacianStdValue = laplacianStdDev.toArray().firstOrNull()?.toFloat() ?: return null
            val pixelCount = grayMat.rows() * grayMat.cols()
            if (pixelCount <= 0) {
                return null
            }

            val grayBytes = ByteArray(pixelCount)
            grayMat.get(0, 0, grayBytes)
            val underexposedCount = grayBytes.count { pixel ->
                (pixel.toInt() and 0xFF) <= DARK_PIXEL_THRESHOLD
            }
            val overexposedCount = grayBytes.count { pixel ->
                (pixel.toInt() and 0xFF) >= BRIGHT_PIXEL_THRESHOLD
            }

            OpenCvFrameImageQualityMetrics(
                meanLuma = meanLuma,
                lumaStdDev = lumaStdDev,
                underexposedRatio = underexposedCount.toFloat() / pixelCount,
                overexposedRatio = overexposedCount.toFloat() / pixelCount,
                laplacianVariance = laplacianStdValue * laplacianStdValue,
            )
        } finally {
            laplacianMean.release()
            laplacianStdDev.release()
        }
    } finally {
        encodedMat.release()
        grayMat.release()
        laplacianMat.release()
        meanMat.release()
        stdDevMat.release()
    }
}

/**
 * Ensures OpenCV native code is ready before metrics are computed.
 *
 * @return true when OpenCV is available in the current process.
 */
private fun isOpenCvReady(): Boolean {
    return openCvReady
}

/**
 * Motion-only metrics used internally by [FrameImageQualityFilter].
 *
 * @param translationDeltaMeters camera translation since previous frame.
 * @param rotationDeltaDegrees camera rotation since previous frame.
 */
private data class MotionMetrics(
    val translationDeltaMeters: Float = 0f,
    val rotationDeltaDegrees: Float = 0f,
)

/**
 * Formats one float value with a fixed number of fractional digits.
 *
 * @param digits number of digits after the decimal separator.
 * @return formatted decimal string.
 */
private fun Float.formatToDigits(digits: Int): String {
    val scale = 10.0.pow(digits.toDouble()).toFloat()
    return ((this * scale).roundToInt() / scale).toString()
}

private val openCvReady: Boolean by lazy {
    runCatching { OpenCVLoader.initDebug() }.getOrDefault(false)
}

private const val DEFAULT_MIN_ACCEPTED_INTERVAL_MS = 50L
private const val DEFAULT_MAX_UNDEREXPOSED_RATIO = 0.35f
private const val DEFAULT_MAX_OVEREXPOSED_RATIO = 0.35f
private const val DEFAULT_MIN_LUMA_STD_DEV = 18f
private const val DEFAULT_MIN_LAPLACIAN_VARIANCE = 120f
private const val DEFAULT_MAX_TRANSLATION_DELTA_METERS = 0.03f
private const val DEFAULT_MAX_ROTATION_DELTA_DEGREES = 8f
private const val DARK_PIXEL_THRESHOLD = 24
private const val BRIGHT_PIXEL_THRESHOLD = 231
