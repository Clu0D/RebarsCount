package anton.axenov

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.FloatBuffer
import javax.imageio.ImageIO
import kotlin.math.ceil
import kotlin.math.floor

/**
 * Creates one JVM ONNX-backed target predictor from a filesystem model path.
 *
 * @param modelSpec static model metadata.
 * @param modelFile filesystem path of the ONNX model.
 * @return ready predictor instance.
 */
fun createJvmOnnxTargetPredictor(
    modelSpec: OnnxModelSpec,
    modelFile: File,
): SegmentationTargetPredictor {
    require(modelFile.isFile) {
        "ONNX model not found for ${modelSpec.modelName}: ${modelFile.path}"
    }
    val runner = JvmOnnxSessionRunner(modelFile)
    return OnnxSegmentationTargetPredictor(
        modelSpec = modelSpec,
        preprocessor = JvmOnnxImageTensorPreprocessor(),
        sessionRunner = runner,
    )
}

/**
 * JVM image preprocessor based on `ImageIO`.
 */
class JvmOnnxImageTensorPreprocessor : OnnxImageTensorPreprocessor {
    /**
     * Decodes, resizes and normalizes one input image.
     *
     * @param imageBytes encoded PNG or JPEG bytes.
     * @param inputDescriptor model input descriptor.
     * @param normalization requested normalization strategy.
     * @return prepared ONNX tensor.
     */
    override fun preprocess(
        imageBytes: ByteArray,
        inputDescriptor: OnnxInputDescriptor,
        normalization: OnnxInputNormalization,
    ): PreparedImageTensor {
        val sourceImage = ImageIO.read(ByteArrayInputStream(imageBytes))
            ?: error("Unsupported image bytes for ONNX predictor")
        val targetWidth = inputDescriptor.width ?: sourceImage.width
        val targetHeight = inputDescriptor.height ?: sourceImage.height
        val resized = resizeImage(sourceImage, targetWidth, targetHeight)
        val values = when (normalization) {
            OnnxInputNormalization.RGB_ZERO_TO_ONE -> rgbZeroToOneInput(
                image = resized,
                channels = inputDescriptor.channels,
                layout = inputDescriptor.layout,
            )

            OnnxInputNormalization.GRAYSCALE_PERCENTILE -> grayscalePercentileInput(
                image = resized,
                channels = inputDescriptor.channels,
                layout = inputDescriptor.layout,
            )
        }
        return PreparedImageTensor(
            originalWidth = sourceImage.width,
            originalHeight = sourceImage.height,
            inputWidth = targetWidth,
            inputHeight = targetHeight,
            channels = inputDescriptor.channels,
            layout = inputDescriptor.layout,
            inputShape = inputShapeFor(
                width = targetWidth,
                height = targetHeight,
                channels = inputDescriptor.channels,
                layout = inputDescriptor.layout,
            ),
            values = values,
        )
    }
}

/**
 * JVM ONNX Runtime session wrapper backed by a model file.
 *
 * @param modelFile filesystem ONNX model file.
 */
class JvmOnnxSessionRunner(
    modelFile: File,
) : OnnxSessionRunner {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = environment.createSession(modelFile.path, OrtSession.SessionOptions())

    /**
     * Decoded model input descriptor.
     */
    override val inputDescriptor: OnnxInputDescriptor = session.toInputDescriptor()

    /**
     * Runs one ONNX inference pass.
     *
     * @param input prepared image tensor.
     * @return all float outputs returned by the model.
     */
    override suspend fun run(input: PreparedImageTensor): List<RawOnnxTensor> {
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input.values), input.inputShape).use { tensor ->
            session.run(mapOf(inputDescriptor.inputName to tensor)).use { results ->
                return results.map { entry ->
                    val outputTensor = entry.value as? OnnxTensor
                        ?: error("ONNX output ${entry.key} is not a float tensor")
                    val tensorInfo = outputTensor.info as? TensorInfo
                        ?: error("ONNX output ${entry.key} does not expose tensor info")
                    RawOnnxTensor(
                        name = entry.key,
                        shape = tensorInfo.shape,
                        values = outputTensor.floatBuffer.toFloatArray(),
                    )
                }
            }
        }
    }

    /**
     * Closes the ONNX session.
     */
    override fun close() {
        session.close()
    }
}

/**
 * Resizes one buffered image to the requested size.
 *
 * @param source source image.
 * @param width target width.
 * @param height target height.
 * @return resized image.
 */
fun resizeImage(
    source: BufferedImage,
    width: Int,
    height: Int,
): BufferedImage {
    if (source.width == width && source.height == height) {
        return source
    }
    return BufferedImage(width, height, BufferedImage.TYPE_INT_RGB).also { target ->
        target.createGraphics().use { graphics ->
            graphics.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR,
            )
            graphics.drawImage(source, 0, 0, width, height, null)
        }
    }
}

/**
 * Builds one input tensor shape.
 *
 * @param width input width.
 * @param height input height.
 * @param channels channel count.
 * @param layout tensor layout.
 * @return ONNX tensor shape.
 */
fun inputShapeFor(
    width: Int,
    height: Int,
    channels: Int,
    layout: OnnxTensorLayout,
): LongArray {
    return when (layout) {
        OnnxTensorLayout.NCHW -> longArrayOf(1, channels.toLong(), height.toLong(), width.toLong())
        OnnxTensorLayout.NHWC -> longArrayOf(1, height.toLong(), width.toLong(), channels.toLong())
    }
}

/**
 * Converts one RGB image to the float input tensor values.
 *
 * @param image resized image.
 * @param channels channel count expected by the model.
 * @param layout tensor layout.
 * @return flattened tensor values.
 */
fun rgbZeroToOneInput(
    image: BufferedImage,
    channels: Int,
    layout: OnnxTensorLayout,
): FloatArray {
    require(channels == 3 || channels == 1) {
        "RGB input supports only one or three channels"
    }
    return when {
        channels == 1 -> grayscaleValues(image)
        layout == OnnxTensorLayout.NHWC -> FloatArray(image.width * image.height * channels) { index ->
            val pixel = index / channels
            val channel = index % channels
            val rgb = image.getRGB(pixel % image.width, pixel / image.width)
            rgbChannel(rgb, channel)
        }

        else -> FloatArray(image.width * image.height * channels) { index ->
            val pixelCount = image.width * image.height
            val channel = index / pixelCount
            val pixel = index % pixelCount
            val rgb = image.getRGB(pixel % image.width, pixel / image.width)
            rgbChannel(rgb, channel)
        }
    }
}

/**
 * Converts one image to percentile-normalized grayscale tensor values.
 *
 * @param image resized image.
 * @param channels channel count expected by the model.
 * @param layout tensor layout.
 * @return flattened tensor values.
 */
fun grayscalePercentileInput(
    image: BufferedImage,
    channels: Int,
    layout: OnnxTensorLayout,
): FloatArray {
    val grayscale = grayscaleValues(image)
    val sorted = grayscale.sorted()
    val low = percentile(sorted, 1.0)
    val high = percentile(sorted, 99.8)
    val scale = (high - low).coerceAtLeast(1e-6f)
    val normalized = grayscale.map { value -> ((value - low) / scale).coerceIn(0f, 1f) }
    return when {
        channels == 1 -> normalized.toFloatArray()
        layout == OnnxTensorLayout.NHWC -> FloatArray(normalized.size * channels) { index ->
            normalized[index / channels]
        }

        else -> FloatArray(normalized.size * channels) { index ->
            normalized[index % normalized.size]
        }
    }
}

/**
 * Extracts grayscale values from one RGB image.
 *
 * @param image source image.
 * @return flattened grayscale values in `0..1`.
 */
fun grayscaleValues(image: BufferedImage): FloatArray {
    return FloatArray(image.width * image.height) { index ->
        val x = index % image.width
        val y = index / image.width
        val rgb = image.getRGB(x, y)
        (0.299f * ((rgb shr 16) and 0xff) +
            0.587f * ((rgb shr 8) and 0xff) +
            0.114f * (rgb and 0xff)) / 255f
    }
}

/**
 * Returns one normalized RGB channel value.
 *
 * @param rgb packed RGB integer.
 * @param channel channel index in `RGB` order.
 * @return normalized `0..1` value.
 */
fun rgbChannel(rgb: Int, channel: Int): Float {
    return when (channel) {
        0 -> ((rgb shr 16) and 0xff) / 255f
        1 -> ((rgb shr 8) and 0xff) / 255f
        else -> (rgb and 0xff) / 255f
    }
}

/**
 * Returns one percentile value from a sorted grayscale list.
 *
 * @param sorted sorted grayscale values.
 * @param percentile percentile in `[0, 100]`.
 * @return interpolated percentile value.
 */
fun percentile(
    sorted: List<Float>,
    percentile: Double,
): Float {
    val index = percentile / 100.0 * (sorted.size - 1)
    val lower = floor(index).toInt()
    val upper = ceil(index).toInt()
    val fraction = (index - lower).toFloat()
    return sorted[lower] * (1f - fraction) + sorted[upper] * fraction
}

/**
 * Converts one float buffer to a detached float array.
 *
 * @return copied float array.
 */
fun FloatBuffer.toFloatArray(): FloatArray {
    rewind()
    return FloatArray(remaining()).also(::get)
}

/**
 * Decodes ONNX input metadata from one loaded session.
 *
 * @return common input descriptor.
 */
fun OrtSession.toInputDescriptor(): OnnxInputDescriptor {
    val inputEntry = inputInfo.entries.singleOrNull()
        ?: error("Expected exactly one ONNX input, found ${inputInfo.keys}")
    val tensorInfo = inputEntry.value.info as? TensorInfo
        ?: error("ONNX input ${inputEntry.key} is not a tensor")
    val layout = inferOnnxInputLayout(tensorInfo.shape)
    return OnnxInputDescriptor(
        inputName = inputEntry.key,
        width = fixedInputDimension(tensorInfo.shape, layout, height = false),
        height = fixedInputDimension(tensorInfo.shape, layout, height = true),
        channels = inputChannels(tensorInfo.shape, layout),
        layout = layout,
    )
}

/**
 * Infers one ONNX input layout from the model input shape.
 *
 * @param shape ONNX input shape.
 * @return inferred layout.
 */
fun inferOnnxInputLayout(shape: LongArray): OnnxTensorLayout {
    require(shape.size == 4) {
        "Expected one rank-4 ONNX image input, got ${shape.contentToString()}"
    }
    return when {
        shape[1] == 1L || shape[1] == 3L -> OnnxTensorLayout.NCHW
        shape[3] == 1L || shape[3] == 3L -> OnnxTensorLayout.NHWC
        else -> error("Cannot infer ONNX input layout from ${shape.contentToString()}")
    }
}

/**
 * Returns one fixed input dimension from a tensor shape.
 *
 * @param shape ONNX tensor shape.
 * @param layout tensor layout.
 * @param height true for height and false for width.
 * @return fixed dimension or null for dynamic models.
 */
fun fixedInputDimension(
    shape: LongArray,
    layout: OnnxTensorLayout,
    height: Boolean,
): Int? {
    val index = when (layout) {
        OnnxTensorLayout.NHWC -> if (height) 1 else 2
        OnnxTensorLayout.NCHW -> if (height) 2 else 3
    }
    return shape[index].takeIf { value -> value > 0 }?.toInt()
}

/**
 * Returns input channel count for one ONNX image shape.
 *
 * @param shape ONNX tensor shape.
 * @param layout tensor layout.
 * @return channel count.
 */
fun inputChannels(
    shape: LongArray,
    layout: OnnxTensorLayout,
): Int {
    return if (layout == OnnxTensorLayout.NHWC) shape[3].toInt() else shape[1].toInt()
}

/**
 * Disposes one AWT graphics instance after use.
 *
 * @param block operation to run.
 */
inline fun <T : java.awt.Graphics> T.use(block: (T) -> Unit) {
    try {
        block(this)
    } finally {
        dispose()
    }
}
