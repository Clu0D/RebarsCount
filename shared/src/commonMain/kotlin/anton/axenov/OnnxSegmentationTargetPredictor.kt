package anton.axenov

import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Real ONNX-backed target predictor.
 *
 * @param modelSpec static model metadata.
 * @param preprocessor platform-specific image preprocessor.
 * @param sessionRunner platform-specific ONNX Runtime wrapper.
 */
class OnnxSegmentationTargetPredictor(
    private val modelSpec: OnnxModelSpec,
    private val preprocessor: OnnxImageTensorPreprocessor,
    private val sessionRunner: OnnxSessionRunner,
) : SegmentationTargetPredictor {
    /**
     * Human-readable predictor name.
     */
    override val predictorName: String = "onnx-${modelSpec.modelName}"

    /**
     * Runs one ONNX inference pass and converts outputs to the shared DTO contract.
     *
     * @param imageBytes encoded PNG or JPEG bytes.
     * @param filename logical image filename.
     * @return segmentation prediction in the same format previously returned by Python.
     */
    override suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
    ): SegmentationPrediction {
        val inputTensor = preprocessor.preprocess(
            imageBytes = imageBytes,
            inputDescriptor = sessionRunner.inputDescriptor,
            normalization = modelSpec.normalization,
        )
        val outputs = sessionRunner.run(inputTensor)
        return when (modelSpec.kind) {
            OnnxSegmentationModelKind.YOLO_SEG -> postprocessYoloSegPrediction(
                modelSpec = modelSpec,
                filename = filename,
                inputTensor = inputTensor,
                outputs = outputs,
            )

            OnnxSegmentationModelKind.YOLO_STARDIST -> postprocessStarDistPrediction(
                modelSpec = modelSpec,
                filename = filename,
                inputTensor = inputTensor,
                outputs = outputs,
            )
        }
    }

    /**
     * Releases the ONNX Runtime session.
     */
    override fun close() {
        sessionRunner.close()
    }
}

/**
 * Converts YOLO segmentation outputs to the shared prediction DTO.
 *
 * @param modelSpec active model metadata.
 * @param filename logical image filename.
 * @param inputTensor prepared input tensor and image dimensions.
 * @param outputs raw ONNX outputs.
 * @return decoded segmentation prediction.
 */
internal fun postprocessYoloSegPrediction(
    modelSpec: OnnxModelSpec,
    filename: String,
    inputTensor: PreparedImageTensor,
    outputs: List<RawOnnxTensor>,
): SegmentationPrediction {
    val (detectionTensor, prototypeTensor) = selectYoloSegOutputs(outputs)
    val prototypes = decodePrototypeTensor(prototypeTensor)
    val candidates = decodeYoloSegCandidates(
        modelSpec = modelSpec,
        inputTensor = inputTensor,
        detectionTensor = detectionTensor,
        maskDimension = prototypes.maskDimension,
    ).sortedByDescending { candidate -> candidate.score }
    val kept = applyBoxNms(candidates, modelSpec.nmsThreshold)
    val instances = kept.mapIndexedNotNull { index, candidate ->
        val polygon = decodeYoloSegPolygon(
            candidate = candidate,
            inputTensor = inputTensor,
            prototypes = prototypes,
            maskThreshold = modelSpec.maskThreshold,
        )
        if (polygon.size < 3) {
            null
        } else {
            val integerPolygon = polygonToImagePoints(
                polygon = polygon,
                fallbackBox = candidate.originalBox,
            )
            SegmentationInstance(
                id = index + 1,
                bbox = boundingBoxFromPolygon(integerPolygon),
                polygon = integerPolygon,
                confidence = candidate.score,
            )
        }
    }
    return SegmentationPrediction(
        filename = filename,
        width = inputTensor.originalWidth,
        height = inputTensor.originalHeight,
        count = instances.size,
        instances = instances,
    )
}

/**
 * Converts StarDist-like ONNX outputs to the shared prediction DTO.
 *
 * @param modelSpec active model metadata.
 * @param filename logical image filename.
 * @param inputTensor prepared input tensor and image dimensions.
 * @param outputs raw ONNX outputs.
 * @return decoded segmentation prediction.
 */
internal fun postprocessStarDistPrediction(
    modelSpec: OnnxModelSpec,
    filename: String,
    inputTensor: PreparedImageTensor,
    outputs: List<RawOnnxTensor>,
): SegmentationPrediction {
    val decoded = decodeStarDistOutputs(outputs, sessionLayout = inputTensor.layout)
    val polygons = StarDistPolygonNms.fromDensePrediction(
        probability = decoded.probability,
        distances = decoded.distances,
        height = decoded.outputHeight,
        width = decoded.outputWidth,
        rays = decoded.rays,
        probabilityThreshold = modelSpec.confidenceThreshold,
        nmsThreshold = modelSpec.nmsThreshold,
        gridY = inferStarDistGrid(inputTensor.inputHeight, decoded.outputHeight),
        gridX = inferStarDistGrid(inputTensor.inputWidth, decoded.outputWidth),
        border = modelSpec.border,
        maxCandidates = modelSpec.maxCandidates,
    )
    val xScale = inputTensor.originalWidth.toFloat() / inputTensor.inputWidth.toFloat()
    val yScale = inputTensor.originalHeight.toFloat() / inputTensor.inputHeight.toFloat()
    val instances = polygons.mapIndexed { index, polygon ->
        val scaledPolygon = polygon.vertices.map { vertex ->
            FloatImagePoint(
                x = (vertex.x * xScale).coerceIn(0f, inputTensor.originalWidth.toFloat()),
                y = (vertex.y * yScale).coerceIn(0f, inputTensor.originalHeight.toFloat()),
            )
        }
        val fallbackBox = polygonBounds(scaledPolygon)
        val integerPolygon = polygonToImagePoints(
            polygon = scaledPolygon,
            fallbackBox = fallbackBox,
        )
        SegmentationInstance(
            id = index + 1,
            bbox = boundingBoxFromPolygon(integerPolygon),
            polygon = integerPolygon,
            confidence = polygon.score,
        )
    }
    return SegmentationPrediction(
        filename = filename,
        width = inputTensor.originalWidth,
        height = inputTensor.originalHeight,
        count = instances.size,
        instances = instances,
    )
}

/**
 * One YOLO segmentation prototype tensor decoded to CHW layout.
 *
 * @param maskDimension number of mask channels.
 * @param height prototype height.
 * @param width prototype width.
 * @param values values stored in CHW order.
 */
internal data class YoloSegPrototypeTensor(
    val maskDimension: Int,
    val height: Int,
    val width: Int,
    val values: FloatArray,
) {
    /**
     * Returns one prototype value from CHW storage.
     *
     * @param channel mask channel index.
     * @param y prototype row.
     * @param x prototype column.
     * @return stored float value.
     */
    fun value(channel: Int, y: Int, x: Int): Float {
        return values[(channel * height + y) * width + x]
    }
}

/**
 * One decoded YOLO segmentation candidate before mask reconstruction.
 *
 * @param score detection confidence.
 * @param originalBox candidate box mapped into original image coordinates.
 * @param modelBox candidate box in model input coordinates.
 * @param maskCoefficients mask coefficients for prototype mixing.
 */
internal data class YoloSegCandidate(
    val score: Float,
    val originalBox: FloatBoundingBox,
    val modelBox: FloatBoundingBox,
    val maskCoefficients: FloatArray,
)

/**
 * One decoded StarDist output bundle.
 *
 * @param probability probability map.
 * @param distances radial distances in HWC layout.
 * @param outputHeight probability-map height.
 * @param outputWidth probability-map width.
 * @param rays number of rays per pixel.
 */
internal data class StarDistDecodedOutputs(
    val probability: FloatArray,
    val distances: FloatArray,
    val outputHeight: Int,
    val outputWidth: Int,
    val rays: Int,
)

/**
 * Selects the detection and prototype outputs of a YOLO segmentation model.
 *
 * @param outputs raw ONNX outputs.
 * @return pair of `(detection, prototypes)` tensors.
 */
internal fun selectYoloSegOutputs(outputs: List<RawOnnxTensor>): Pair<RawOnnxTensor, RawOnnxTensor> {
    val prototypeTensor = outputs.firstOrNull { tensor -> tensor.shape.size == 4 }
        ?: error("YOLO segmentation model must return one rank-4 prototype tensor")
    val detectionTensor = outputs.firstOrNull { tensor ->
        tensor !== prototypeTensor && (tensor.shape.size == 2 || tensor.shape.size == 3)
    } ?: error("YOLO segmentation model must return one detection tensor")
    return detectionTensor to prototypeTensor
}

/**
 * Decodes one YOLO prototype tensor into CHW layout.
 *
 * @param tensor raw prototype tensor.
 * @return decoded CHW tensor.
 */
internal fun decodePrototypeTensor(tensor: RawOnnxTensor): YoloSegPrototypeTensor {
    require(tensor.shape.size == 4) {
        "YOLO prototype tensor must be rank-4"
    }
    val shape = tensor.shape
    val isNchw = shape[1] <= shape[2] && shape[1] <= shape[3]
    return if (isNchw) {
        YoloSegPrototypeTensor(
            maskDimension = shape[1].toInt(),
            height = shape[2].toInt(),
            width = shape[3].toInt(),
            values = tensor.values,
        )
    } else {
        val height = shape[1].toInt()
        val width = shape[2].toInt()
        val channels = shape[3].toInt()
        val chw = FloatArray(height * width * channels)
        for (y in 0 until height) {
            for (x in 0 until width) {
                for (channel in 0 until channels) {
                    chw[(channel * height + y) * width + x] =
                        tensor.values[(y * width + x) * channels + channel]
                }
            }
        }
        YoloSegPrototypeTensor(
            maskDimension = channels,
            height = height,
            width = width,
            values = chw,
        )
    }
}

/**
 * Decodes detector candidates from a YOLO segmentation output tensor.
 *
 * @param modelSpec active model metadata.
 * @param inputTensor prepared input tensor.
 * @param detectionTensor raw detection tensor.
 * @param maskDimension number of mask coefficients appended to each detection.
 * @return retained candidates before NMS.
 */
internal fun decodeYoloSegCandidates(
    modelSpec: OnnxModelSpec,
    inputTensor: PreparedImageTensor,
    detectionTensor: RawOnnxTensor,
    maskDimension: Int,
): List<YoloSegCandidate> {
    val attributes = detectionAttributes(detectionTensor.shape)
    val count = detectionCount(detectionTensor.shape)
    val rows = ArrayList<FloatArray>(count)
    repeat(count) { detectionIndex ->
        rows += FloatArray(attributes) { attributeIndex ->
            detectionValue(detectionTensor, detectionIndex, attributeIndex, count, attributes)
        }
    }
    val xScale = inputTensor.originalWidth.toFloat() / inputTensor.inputWidth.toFloat()
    val yScale = inputTensor.originalHeight.toFloat() / inputTensor.inputHeight.toFloat()
    return rows.mapNotNull { row ->
        val layout = inferYoloSegFieldLayout(
            modelSpec = modelSpec,
            attributeCount = row.size,
            maskDimension = maskDimension,
        )
        val confidence = yoloSegConfidence(row, layout)
        if (confidence < modelSpec.confidenceThreshold) {
            return@mapNotNull null
        }
        val cx = denormalizeCoordinate(row[0], inputTensor.inputWidth.toFloat())
        val cy = denormalizeCoordinate(row[1], inputTensor.inputHeight.toFloat())
        val width = denormalizeCoordinate(row[2], inputTensor.inputWidth.toFloat())
        val height = denormalizeCoordinate(row[3], inputTensor.inputHeight.toFloat())
        val modelBox = FloatBoundingBox(
            left = cx - width / 2f,
            top = cy - height / 2f,
            right = cx + width / 2f,
            bottom = cy + height / 2f,
        ).clipped(inputTensor.inputWidth, inputTensor.inputHeight)
        val originalBox = FloatBoundingBox(
            left = modelBox.left * xScale,
            top = modelBox.top * yScale,
            right = modelBox.right * xScale,
            bottom = modelBox.bottom * yScale,
        ).clipped(inputTensor.originalWidth, inputTensor.originalHeight)
        YoloSegCandidate(
            score = confidence,
            originalBox = originalBox,
            modelBox = modelBox,
            maskCoefficients = row.copyOfRange(layout.maskStart, layout.maskStart + maskDimension),
        )
    }
}

/**
 * Applies standard bounding-box NMS to YOLO segmentation candidates.
 *
 * @param candidates sorted or unsorted candidates.
 * @param threshold maximal allowed IoU.
 * @return retained candidates sorted by descending score.
 */
internal fun applyBoxNms(
    candidates: List<YoloSegCandidate>,
    threshold: Float,
): List<YoloSegCandidate> {
    val sorted = candidates.sortedByDescending { candidate -> candidate.score }
    val kept = mutableListOf<YoloSegCandidate>()
    sorted.forEach { candidate ->
        if (kept.none { accepted -> boundingBoxIou(accepted.originalBox, candidate.originalBox) > threshold }) {
            kept += candidate
        }
    }
    return kept
}

/**
 * Reconstructs one polygon from YOLO prototype masks.
 *
 * @param candidate retained detection candidate.
 * @param inputTensor prepared input tensor.
 * @param prototypes prototype tensor in CHW layout.
 * @param maskThreshold mask probability threshold.
 * @return polygon vertices in original image coordinates.
 */
internal fun decodeYoloSegPolygon(
    candidate: YoloSegCandidate,
    inputTensor: PreparedImageTensor,
    prototypes: YoloSegPrototypeTensor,
    maskThreshold: Float,
): List<FloatImagePoint> {
    val protoLeft = floor(candidate.modelBox.left / inputTensor.inputWidth * prototypes.width).toInt()
        .coerceIn(0, prototypes.width - 1)
    val protoTop = floor(candidate.modelBox.top / inputTensor.inputHeight * prototypes.height).toInt()
        .coerceIn(0, prototypes.height - 1)
    val protoRight = floor(candidate.modelBox.right / inputTensor.inputWidth * prototypes.width).toInt()
        .coerceIn(protoLeft, prototypes.width - 1)
    val protoBottom = floor(candidate.modelBox.bottom / inputTensor.inputHeight * prototypes.height).toInt()
        .coerceIn(protoTop, prototypes.height - 1)
    val foreground = mutableListOf<FloatImagePoint>()
    for (y in protoTop..protoBottom) {
        for (x in protoLeft..protoRight) {
            var mixedValue = 0f
            for (channel in candidate.maskCoefficients.indices) {
                mixedValue += candidate.maskCoefficients[channel] * prototypes.value(channel, y, x)
            }
            if (sigmoid(mixedValue) >= maskThreshold) {
                foreground += FloatImagePoint(
                    x = ((x + 0.5f) / prototypes.width) * inputTensor.originalWidth,
                    y = ((y + 0.5f) / prototypes.height) * inputTensor.originalHeight,
                )
            }
        }
    }
    return polygonFromForegroundPoints(
        points = foreground,
        fallbackBox = candidate.originalBox,
    )
}

/**
 * Decodes StarDist outputs from either split or merged ONNX tensors.
 *
 * @param outputs raw ONNX outputs.
 * @param sessionLayout input layout used by the model as a fallback for ambiguous outputs.
 * @return decoded StarDist tensors.
 */
internal fun decodeStarDistOutputs(
    outputs: List<RawOnnxTensor>,
    sessionLayout: OnnxTensorLayout,
): StarDistDecodedOutputs {
    val rankFourOutputs = outputs.filter { tensor -> tensor.shape.size == 4 && tensor.shape[0] == 1L }
    require(rankFourOutputs.isNotEmpty()) {
        "StarDist ONNX model must return at least one rank-4 output"
    }
    val probabilityTensor = rankFourOutputs.firstOrNull { tensor ->
        inferredOutputChannels(tensor.shape, sessionLayout) == 1
    }
    if (probabilityTensor != null) {
        val probabilityLayout = inferOutputTensorLayout(probabilityTensor.shape, sessionLayout)
        val probabilityChannelTensor = toChannelTensor(probabilityTensor, probabilityLayout)
        val distanceTensor = rankFourOutputs.firstOrNull { tensor ->
            tensor !== probabilityTensor &&
                inferredOutputChannels(tensor.shape, inferOutputTensorLayout(tensor.shape, probabilityLayout)) > 1
        } ?: error("StarDist ONNX model did not provide a distance tensor")
        val distanceChannelTensor = toChannelTensor(
            distanceTensor,
            inferOutputTensorLayout(distanceTensor.shape, probabilityLayout),
        )
        return StarDistDecodedOutputs(
            probability = probabilityChannelTensor.channel(0),
            distances = distanceChannelTensor.channels(0),
            outputHeight = probabilityChannelTensor.height,
            outputWidth = probabilityChannelTensor.width,
            rays = distanceChannelTensor.channels,
        )
    }
    val mergedTensor = rankFourOutputs
        .map { tensor -> toChannelTensor(tensor, inferOutputTensorLayout(tensor.shape, sessionLayout)) }
        .firstOrNull { tensor -> tensor.channels > 1 }
        ?: error("StarDist ONNX model did not provide probability and distance channels")
    return StarDistDecodedOutputs(
        probability = mergedTensor.channel(0),
        distances = mergedTensor.channels(1),
        outputHeight = mergedTensor.height,
        outputWidth = mergedTensor.width,
        rays = mergedTensor.channels - 1,
    )
}

/**
 * Simple HWC tensor wrapper used by StarDist output decoding.
 *
 * @param height tensor height.
 * @param width tensor width.
 * @param channels tensor channel count.
 * @param values values stored in HWC layout.
 */
internal data class ChannelTensor(
    val height: Int,
    val width: Int,
    val channels: Int,
    val values: FloatArray,
) {
    /**
     * Returns one channel as a flattened HW array.
     *
     * @param channelIndex selected channel.
     * @return flattened HW channel values.
     */
    fun channel(channelIndex: Int): FloatArray {
        return FloatArray(height * width) { index -> values[index * channels + channelIndex] }
    }

    /**
     * Returns all channels from one starting index as a flattened HWC array.
     *
     * @param firstChannel first channel to copy.
     * @return flattened HWC values.
     */
    fun channels(firstChannel: Int): FloatArray {
        val remainingChannels = channels - firstChannel
        return FloatArray(height * width * remainingChannels) { index ->
            val pixel = index / remainingChannels
            val channel = index % remainingChannels
            values[pixel * channels + firstChannel + channel]
        }
    }
}

/**
 * Converts one ONNX output tensor to HWC layout.
 *
 * @param tensor raw ONNX output tensor.
 * @param layout inferred tensor layout.
 * @return HWC channel tensor.
 */
internal fun toChannelTensor(
    tensor: RawOnnxTensor,
    layout: OnnxTensorLayout,
): ChannelTensor {
    val shape = tensor.shape
    val height = if (layout == OnnxTensorLayout.NHWC) shape[1].toInt() else shape[2].toInt()
    val width = if (layout == OnnxTensorLayout.NHWC) shape[2].toInt() else shape[3].toInt()
    val channels = inferredOutputChannels(shape, layout)
    if (layout == OnnxTensorLayout.NHWC) {
        return ChannelTensor(
            height = height,
            width = width,
            channels = channels,
            values = tensor.values,
        )
    }
    val hwc = FloatArray(height * width * channels)
    for (channel in 0 until channels) {
        for (y in 0 until height) {
            for (x in 0 until width) {
                hwc[(y * width + x) * channels + channel] =
                    tensor.values[(channel * height + y) * width + x]
            }
        }
    }
    return ChannelTensor(
        height = height,
        width = width,
        channels = channels,
        values = hwc,
    )
}

/**
 * Returns one polygon bounds box.
 *
 * @param polygon polygon vertices.
 * @return covering float box.
 */
internal fun polygonBounds(polygon: List<FloatImagePoint>): FloatBoundingBox {
    if (polygon.isEmpty()) {
        return FloatBoundingBox(0f, 0f, 0f, 0f)
    }
    return FloatBoundingBox(
        left = polygon.minOf { point -> point.x },
        top = polygon.minOf { point -> point.y },
        right = polygon.maxOf { point -> point.x },
        bottom = polygon.maxOf { point -> point.y },
    )
}

/**
 * Returns one safe sigmoid value.
 *
 * @param value raw logit.
 * @return sigmoid probability in range `[0, 1]`.
 */
internal fun sigmoid(value: Float): Float {
    return (1f / (1f + exp(-value)))
}

/**
 * Returns detector attribute count from one YOLO output shape.
 *
 * @param shape output shape.
 * @return attribute count.
 */
internal fun detectionAttributes(shape: LongArray): Int {
    return when (shape.size) {
        2 -> selectDetectionAttributes(shape[0].toInt(), shape[1].toInt())
        3 -> selectDetectionAttributes(shape[1].toInt(), shape[2].toInt())
        else -> error("Unsupported YOLO detection tensor rank: ${shape.size}")
    }
}

/**
 * Returns detector candidate count from one YOLO output shape.
 *
 * @param shape output shape.
 * @return number of detector rows.
 */
internal fun detectionCount(shape: LongArray): Int {
    return when (shape.size) {
        2 -> selectDetectionCount(shape[0].toInt(), shape[1].toInt())
        3 -> selectDetectionCount(shape[1].toInt(), shape[2].toInt())
        else -> error("Unsupported YOLO detection tensor rank: ${shape.size}")
    }
}

/**
 * Chooses which of two dimensions represents detector attributes.
 *
 * This handles the common degenerate case where the detector returns exactly
 * one candidate and the second dimension therefore equals `1`.
 *
 * @param first first non-batch dimension.
 * @param second second non-batch dimension.
 * @return selected attribute count.
 */
internal fun selectDetectionAttributes(
    first: Int,
    second: Int,
): Int {
    return when {
        first <= 4 && second > 4 -> second
        second <= 4 && first > 4 -> first
        first <= second -> first
        else -> second
    }
}

/**
 * Chooses which of two dimensions represents detector candidate count.
 *
 * @param first first non-batch dimension.
 * @param second second non-batch dimension.
 * @return selected candidate count.
 */
internal fun selectDetectionCount(
    first: Int,
    second: Int,
): Int {
    return if (selectDetectionAttributes(first, second) == first) second else first
}

/**
 * Reads one detector value regardless of whether the tensor is transposed.
 *
 * @param tensor raw detection tensor.
 * @param detectionIndex detection row index.
 * @param attributeIndex attribute column index.
 * @param detectionCount expected detection count.
 * @param attributeCount expected attribute count.
 * @return stored float value.
 */
internal fun detectionValue(
    tensor: RawOnnxTensor,
    detectionIndex: Int,
    attributeIndex: Int,
    detectionCount: Int,
    attributeCount: Int,
): Float {
    val shape = tensor.shape
    return when (shape.size) {
        2 -> {
            if (shape[0].toInt() == detectionCount) {
                tensor.values[detectionIndex * attributeCount + attributeIndex]
            } else {
                tensor.values[attributeIndex * detectionCount + detectionIndex]
            }
        }

        3 -> {
            if (shape[1].toInt() == detectionCount) {
                tensor.values[detectionIndex * attributeCount + attributeIndex]
            } else {
                tensor.values[attributeIndex * detectionCount + detectionIndex]
            }
        }

        else -> error("Unsupported YOLO detection tensor rank: ${shape.size}")
    }
}

/**
 * Logical layout of class and mask fields inside one YOLO detection row.
 *
 * @param objectnessIndex optional objectness index.
 * @param classStart index of the first class score.
 * @param classCount number of class scores.
 * @param maskStart index of the first mask coefficient.
 */
internal data class YoloSegFieldLayout(
    val objectnessIndex: Int?,
    val classStart: Int,
    val classCount: Int,
    val maskStart: Int,
)

/**
 * Infers field positions inside one YOLO segmentation detection row.
 *
 * @param modelSpec active model metadata.
 * @param attributeCount total attribute count.
 * @param maskDimension prototype channel count.
 * @return field layout.
 */
internal fun inferYoloSegFieldLayout(
    modelSpec: OnnxModelSpec,
    attributeCount: Int,
    maskDimension: Int,
): YoloSegFieldLayout {
    val remainingAfterBox = attributeCount - 4
    require(remainingAfterBox > maskDimension) {
        "YOLO segmentation row does not contain class scores and mask coefficients"
    }
    val expectedClasses = modelSpec.expectedClassCount
    return when {
        remainingAfterBox == expectedClasses + maskDimension -> YoloSegFieldLayout(
            objectnessIndex = null,
            classStart = 4,
            classCount = expectedClasses,
            maskStart = 4 + expectedClasses,
        )

        remainingAfterBox == 1 + expectedClasses + maskDimension -> YoloSegFieldLayout(
            objectnessIndex = 4,
            classStart = 5,
            classCount = expectedClasses,
            maskStart = 5 + expectedClasses,
        )

        else -> YoloSegFieldLayout(
            objectnessIndex = null,
            classStart = 4,
            classCount = remainingAfterBox - maskDimension,
            maskStart = attributeCount - maskDimension,
        )
    }
}

/**
 * Returns one detection confidence from a decoded field layout.
 *
 * @param row raw detector row.
 * @param layout decoded field layout.
 * @return confidence score in range `[0, 1]`.
 */
internal fun yoloSegConfidence(
    row: FloatArray,
    layout: YoloSegFieldLayout,
): Float {
    val classScore = row
        .copyOfRange(layout.classStart, layout.classStart + layout.classCount)
        .maxOrNull()
        ?: 0f
    val objectness = layout.objectnessIndex?.let { index -> row[index] } ?: 1f
    return (classScore * objectness).coerceIn(0f, 1f)
}

/**
 * Denormalizes one detector coordinate when the model output uses `[0, 1]`.
 *
 * @param value raw coordinate value.
 * @param axisSize input axis size in pixels.
 * @return coordinate in model input pixels.
 */
internal fun denormalizeCoordinate(
    value: Float,
    axisSize: Float,
): Float {
    return if (value in 0f..1.5f) value * axisSize else value
}

/**
 * Computes IoU for two axis-aligned boxes.
 *
 * @param first first box.
 * @param second second box.
 * @return IoU in range `[0, 1]`.
 */
internal fun boundingBoxIou(
    first: FloatBoundingBox,
    second: FloatBoundingBox,
): Float {
    val intersectionLeft = max(first.left, second.left)
    val intersectionTop = max(first.top, second.top)
    val intersectionRight = min(first.right, second.right)
    val intersectionBottom = min(first.bottom, second.bottom)
    val intersection = FloatBoundingBox(
        left = intersectionLeft,
        top = intersectionTop,
        right = intersectionRight,
        bottom = intersectionBottom,
    )
    val union = first.area() + second.area() - intersection.area()
    if (union <= 0f) {
        return 0f
    }
    return (intersection.area() / union).coerceIn(0f, 1f)
}

/**
 * Infers layout of one rank-4 ONNX output tensor.
 *
 * @param shape tensor shape.
 * @param fallback layout used when the output is ambiguous.
 * @return inferred layout.
 */
internal fun inferOutputTensorLayout(
    shape: LongArray,
    fallback: OnnxTensorLayout,
): OnnxTensorLayout {
    return when {
        shape[1] == 1L -> OnnxTensorLayout.NCHW
        shape[3] == 1L -> OnnxTensorLayout.NHWC
        shape[1] < shape[3] -> OnnxTensorLayout.NCHW
        shape[3] < shape[1] -> OnnxTensorLayout.NHWC
        else -> fallback
    }
}

/**
 * Returns channel count of one rank-4 tensor for the given layout.
 *
 * @param shape tensor shape.
 * @param layout tensor layout.
 * @return channel count.
 */
internal fun inferredOutputChannels(
    shape: LongArray,
    layout: OnnxTensorLayout,
): Int {
    return if (layout == OnnxTensorLayout.NHWC) shape[3].toInt() else shape[1].toInt()
}
