package anton.axenov

import kotlin.math.max
import kotlin.math.min

/**
 * Tensor channel layout used by ONNX image models.
 */
enum class OnnxTensorLayout {
    NCHW,
    NHWC,
}

/**
 * Input normalization strategy required by one ONNX model.
 */
enum class OnnxInputNormalization {
    RGB_ZERO_TO_ONE,
    GRAYSCALE_PERCENTILE,
}

/**
 * High-level output family produced by one ONNX segmentation model.
 */
enum class OnnxSegmentationModelKind {
    YOLO_SEG,
    YOLO_STARDIST,
}

/**
 * Decoded ONNX input tensor contract discovered from one model session.
 *
 * @param inputName logical ONNX input name.
 * @param width model input width, or null when dynamic.
 * @param height model input height, or null when dynamic.
 * @param channels input channel count.
 * @param layout tensor layout expected by the model.
 */
data class OnnxInputDescriptor(
    val inputName: String,
    val width: Int?,
    val height: Int?,
    val channels: Int,
    val layout: OnnxTensorLayout,
)

/**
 * Static metadata for one repository-managed ONNX model.
 *
 * @param modelName human-readable model identifier.
 * @param target segmentation task served by the model.
 * @param kind output family produced by the model.
 * @param relativePath path relative to the repository model directory.
 * @param normalization image normalization strategy.
 * @param confidenceThreshold minimal confidence for retaining one detection.
 * @param nmsThreshold NMS overlap threshold used by the postprocessor.
 * @param maskThreshold mask probability threshold for polygon extraction.
 * @param expectedClassCount expected number of object classes in detector output.
 * @param border ignored border width for StarDist dense predictions.
 * @param maxCandidates maximal number of StarDist candidates processed after sorting.
 */
data class OnnxModelSpec(
    val modelName: String,
    val target: SegmentationPredictionTarget,
    val kind: OnnxSegmentationModelKind,
    val relativePath: String,
    val normalization: OnnxInputNormalization,
    val confidenceThreshold: Float,
    val nmsThreshold: Float,
    val maskThreshold: Float = 0.5f,
    val expectedClassCount: Int = 1,
    val border: Int = 2,
    val maxCandidates: Int = 10_000,
)

/**
 * Prepared image tensor ready to be fed into one ONNX model.
 *
 * @param originalWidth original decoded image width.
 * @param originalHeight original decoded image height.
 * @param inputWidth width used for the input tensor.
 * @param inputHeight height used for the input tensor.
 * @param channels input channel count.
 * @param layout tensor layout used by [values].
 * @param inputShape full ONNX tensor shape.
 * @param values normalized input tensor values.
 */
data class PreparedImageTensor(
    val originalWidth: Int,
    val originalHeight: Int,
    val inputWidth: Int,
    val inputHeight: Int,
    val channels: Int,
    val layout: OnnxTensorLayout,
    val inputShape: LongArray,
    val values: FloatArray,
)

/**
 * One raw float tensor returned from ONNX Runtime.
 *
 * @param name logical ONNX output name.
 * @param shape tensor shape.
 * @param values flattened float values.
 */
data class RawOnnxTensor(
    val name: String,
    val shape: LongArray,
    val values: FloatArray,
)

/**
 * Platform-specific image preprocessor used by ONNX predictors.
 */
interface OnnxImageTensorPreprocessor {
    /**
     * Decodes and normalizes one input image for the target model.
     *
     * @param imageBytes encoded PNG or JPEG bytes.
     * @param inputDescriptor model input descriptor.
     * @param normalization requested normalization strategy.
     * @return prepared input tensor and original image dimensions.
     */
    fun preprocess(
        imageBytes: ByteArray,
        inputDescriptor: OnnxInputDescriptor,
        normalization: OnnxInputNormalization,
    ): PreparedImageTensor
}

/**
 * Platform-specific ONNX Runtime session wrapper.
 */
interface OnnxSessionRunner {
    /**
     * Decoded ONNX input descriptor for the loaded model.
     */
    val inputDescriptor: OnnxInputDescriptor

    /**
     * Runs one ONNX inference pass.
     *
     * @param input prepared image tensor.
     * @return all float outputs returned by the session.
     */
    suspend fun run(input: PreparedImageTensor): List<RawOnnxTensor>

    /**
     * Releases all runtime resources owned by the session.
     */
    fun close()
}

/**
 * Repository layout for ONNX model files.
 */
object OnnxModelRepositoryLayout {
    const val MODEL_DIRECTORY = "models/onnx"
    const val ZONES_YOLO_SEG_MODEL = "zones_yolo_seg.onnx"
    const val POINTS_YOLO_SEG_MODEL = "points_yolo_seg.onnx"
    const val POINTS_YOLO_STARDIST_MODEL = "points_yolo_stardist.onnx"
}

/**
 * Built-in model catalog used by local and server predictor factories.
 */
object OnnxModelCatalog {
    val zonesYoloSeg: OnnxModelSpec = OnnxModelSpec(
        modelName = "zones-yolo-seg",
        target = SegmentationPredictionTarget.ZONES,
        kind = OnnxSegmentationModelKind.YOLO_SEG,
        relativePath = "${OnnxModelRepositoryLayout.MODEL_DIRECTORY}/${OnnxModelRepositoryLayout.ZONES_YOLO_SEG_MODEL}",
        normalization = OnnxInputNormalization.RGB_ZERO_TO_ONE,
        confidenceThreshold = 0.25f,
        nmsThreshold = 0.45f,
        maskThreshold = 0.5f,
        expectedClassCount = 1,
    )

    val pointsYoloSeg: OnnxModelSpec = OnnxModelSpec(
        modelName = "points-yolo-seg",
        target = SegmentationPredictionTarget.POINTS,
        kind = OnnxSegmentationModelKind.YOLO_SEG,
        relativePath = "${OnnxModelRepositoryLayout.MODEL_DIRECTORY}/${OnnxModelRepositoryLayout.POINTS_YOLO_SEG_MODEL}",
        normalization = OnnxInputNormalization.RGB_ZERO_TO_ONE,
        confidenceThreshold = 0.20f,
        nmsThreshold = 0.40f,
        maskThreshold = 0.5f,
        expectedClassCount = 1,
    )

    val pointsYoloStarDist: OnnxModelSpec = OnnxModelSpec(
        modelName = "points-yolo-stardist",
        target = SegmentationPredictionTarget.POINTS,
        kind = OnnxSegmentationModelKind.YOLO_STARDIST,
        relativePath = "${OnnxModelRepositoryLayout.MODEL_DIRECTORY}/${OnnxModelRepositoryLayout.POINTS_YOLO_STARDIST_MODEL}",
        normalization = OnnxInputNormalization.GRAYSCALE_PERCENTILE,
        confidenceThreshold = 0.50f,
        nmsThreshold = 0.50f,
        expectedClassCount = 1,
        border = 2,
        maxCandidates = 10_000,
    )
}

/**
 * Float polygon point used by common ONNX post-processing.
 *
 * @param x X coordinate in image space.
 * @param y Y coordinate in image space.
 */
data class FloatImagePoint(
    val x: Float,
    val y: Float,
)

/**
 * Float bounding box in XYXY representation.
 *
 * @param left left boundary in pixels.
 * @param top top boundary in pixels.
 * @param right right boundary in pixels.
 * @param bottom bottom boundary in pixels.
 */
data class FloatBoundingBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    /**
     * Returns the width of the box.
     *
     * @return non-negative width.
     */
    fun width(): Float = max(0f, right - left)

    /**
     * Returns the height of the box.
     *
     * @return non-negative height.
     */
    fun height(): Float = max(0f, bottom - top)

    /**
     * Returns the area of the box.
     *
     * @return non-negative area.
     */
    fun area(): Float = width() * height()

    /**
     * Clips the box to one image rectangle.
     *
     * @param width image width.
     * @param height image height.
     * @return clipped box.
     */
    fun clipped(width: Int, height: Int): FloatBoundingBox {
        return FloatBoundingBox(
            left = left.coerceIn(0f, width.toFloat()),
            top = top.coerceIn(0f, height.toFloat()),
            right = right.coerceIn(0f, width.toFloat()),
            bottom = bottom.coerceIn(0f, height.toFloat()),
        )
    }
}

/**
 * Converts one float polygon to the shared DTO representation.
 *
 * @param polygon float polygon vertices.
 * @param fallbackBox fallback box used when the polygon degenerates.
 * @return polygon with integer coordinates and at least three points.
 */
internal fun polygonToImagePoints(
    polygon: List<FloatImagePoint>,
    fallbackBox: FloatBoundingBox,
): List<ImagePoint> {
    val integerPoints = polygon
        .map { point ->
            ImagePoint(
                x = point.x.toInt(),
                y = point.y.toInt(),
            )
        }
        .distinct()
    if (integerPoints.size >= 3) {
        return integerPoints
    }
    val left = min(fallbackBox.left, fallbackBox.right).toInt()
    val top = min(fallbackBox.top, fallbackBox.bottom).toInt()
    val right = max(max(fallbackBox.left, fallbackBox.right).toInt(), left + 1)
    val bottom = max(max(fallbackBox.top, fallbackBox.bottom).toInt(), top + 1)
    return listOf(
        ImagePoint(left, top),
        ImagePoint(right, top),
        ImagePoint(right, bottom),
        ImagePoint(left, bottom),
    )
}

/**
 * Builds one segmentation bounding box from a polygon.
 *
 * @param polygon polygon vertices in integer image coordinates.
 * @return axis-aligned box that covers the polygon.
 */
internal fun boundingBoxFromPolygon(polygon: List<ImagePoint>): SegmentationBoundingBox {
    val minX = polygon.minOf { point -> point.x }
    val minY = polygon.minOf { point -> point.y }
    val maxX = polygon.maxOf { point -> point.x }
    val maxY = polygon.maxOf { point -> point.y }
    return SegmentationBoundingBox(
        x = minX,
        y = minY,
        width = (maxX - minX).coerceAtLeast(1),
        height = (maxY - minY).coerceAtLeast(1),
    )
}
