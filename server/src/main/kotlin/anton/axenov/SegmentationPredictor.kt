package anton.axenov

import io.ktor.client.HttpClient
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

/**
 * JVM predictor that can serve either one fixed target or the whole provider contract.
 *
 * @param baseUrl Python server base URL.
 * @param target optional fixed segmentation target served by this predictor instance.
 * @param httpClient HTTP client used for multipart requests.
 */
class SegmentationPredictor(
    private val baseUrl: String = PYTHON_SEGMENTATION_SERVER_URL,
    private val target: SegmentationPredictionTarget? = null,
    private val httpClient: HttpClient = HttpClient {
        install(ContentNegotiation) {
            json(
                Json {
                    ignoreUnknownKeys = true
                },
            )
        }
    },
) : SegmentationTargetPredictor, SegmentationPredictionProvider {
    /**
     * Human-readable predictor name.
     */
    override val predictorName: String = "python-${(target ?: SegmentationPredictionTarget.ZONES).name.lowercase()}"

    /**
     * Sends one image to Python predictor.
     *
     * @param imageBytes PNG bytes to predict.
     * @param filename logical source filename.
     * @return segmentation prediction result.
     */
    override suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
    ): SegmentationPrediction {
        val effectiveTarget = requireNotNull(target) {
            "Target-specific predict(imageBytes, filename) requires a fixed segmentation target"
        }
        val url = when (effectiveTarget) {
            SegmentationPredictionTarget.ZONES -> "$baseUrl/predict_zones"
            SegmentationPredictionTarget.POINTS -> "$baseUrl/predict_points"
        }
        val response = httpClient.submitFormWithBinaryData(
            url = url,
            formData = formData {
                append(
                    key = "file",
                    value = imageBytes,
                    headers = Headers.build {
                        append(
                            HttpHeaders.ContentDisposition,
                            "form-data; name=\"file\"; filename=\"$filename\"",
                        )
                        append(HttpHeaders.ContentType, ContentType.Image.PNG.toString())
                    },
                )
            },
        )
        if (!response.status.isSuccess()) {
            error("Python segmentation failed: ${response.status.value} ${response.bodyAsText()}")
        }
        val responseBody = response.bodyAsText()
        return try {
            predictorJson.decodeFromString(responseBody)
        } catch (error: Exception) {
            throw IllegalStateException(
                "Python segmentation returned invalid JSON: " +
                        responseBody.take(MAX_PREDICTOR_ERROR_BODY_LENGTH),
                error,
            )
        }
    }

    /**
     * Sends one image to the Python predictor selected by request kind.
     *
     * @param imageBytes PNG bytes to predict.
     * @param filename logical source filename.
     * @param zonePrediction true for zone detection and false for point detection.
     * @return segmentation prediction result.
     */
    override suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
        zonePrediction: Boolean,
    ): SegmentationPrediction {
        val effectiveTarget = if (zonePrediction) {
            SegmentationPredictionTarget.ZONES
        } else {
            SegmentationPredictionTarget.POINTS
        }
        val targetPredictor = if (target == null || target == effectiveTarget) {
            this
        } else {
            SegmentationPredictor(
                baseUrl = baseUrl,
                target = effectiveTarget,
                httpClient = httpClient,
            )
        }
        return targetPredictor.predict(
            imageBytes = imageBytes,
            filename = filename,
        )
    }

    /**
     * Releases owned HTTP client resources.
     */
    override fun close() {
        httpClient.close()
    }
}

private val predictorJson = Json {
    ignoreUnknownKeys = true
}
private const val MAX_PREDICTOR_ERROR_BODY_LENGTH = 2_000
