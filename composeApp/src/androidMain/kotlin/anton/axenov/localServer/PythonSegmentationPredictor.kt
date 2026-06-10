package anton.axenov.localServer

import anton.axenov.SegmentationPrediction
import anton.axenov.SegmentationPredictionProvider
import io.ktor.client.HttpClient
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import io.ktor.http.isSuccess
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

/**
 * Temporary Android prediction adapter that forwards images directly to Python.
 *
 * @param baseUrl Python service base URL.
 * @param httpClient HTTP client used for multipart requests.
 */
class PythonSegmentationPredictor(
    baseUrl: String,
    private val httpClient: HttpClient,
) : SegmentationPredictionProvider {
    private val normalizedBaseUrl = baseUrl.trimEnd('/')
    private val json = Json {
        ignoreUnknownKeys = true
    }

    /**
     * Sends one image directly to the temporary Python prediction service.
     *
     * @param imageBytes PNG image bytes.
     * @param filename logical image filename.
     * @param zonePrediction true for zone detection and false for point detection.
     * @return decoded segmentation prediction.
     */
    override suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
        zonePrediction: Boolean,
    ): SegmentationPrediction {
        val endpoint = if (zonePrediction) "predict_zones" else "predict_points"
        val response = httpClient.submitFormWithBinaryData(
            url = "$normalizedBaseUrl/$endpoint",
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
        val responseBody = response.bodyAsText()
        if (!response.status.isSuccess()) {
            throw IllegalStateException(
                "Python segmentation failed: HTTP ${response.status.value}: " +
                    responseBody.take(MAX_ERROR_BODY_LENGTH),
            )
        }
        return try {
            json.decodeFromString(responseBody)
        } catch (error: Exception) {
            throw IllegalStateException(
                "Python segmentation returned invalid JSON: " +
                    responseBody.take(MAX_ERROR_BODY_LENGTH),
                error,
            )
        }
    }

    /**
     * Releases the HTTP client.
     */
    override fun close() {
        httpClient.close()
    }
}

private const val MAX_ERROR_BODY_LENGTH = 2_000
