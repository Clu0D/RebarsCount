package anton.axenov

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json


/**
 * Predicts segmentation for one image by calling external Python service.
 *
 * @param baseUrl Python server base URL.
 * @param httpClient HTTP client used for multipart requests.
 */
class SegmentationPredictor(
    private val baseUrl: String = PYTHON_SEGMENTATION_SERVER_URL,
    private val httpClient: HttpClient = HttpClient {
        install(ContentNegotiation) {
            json(
                Json {
                    ignoreUnknownKeys = true
                },
            )
        }
    },
) {
    /**
     * Sends one image to Python predictor.
     *
     * @param imageBytes PNG bytes to predict.
     * @param filename logical source filename.
     * @return segmentation prediction result.
     */
    suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
        zonePrediction:Boolean,
    ): SegmentationPrediction {
        val url = if (zonePrediction)
            "$baseUrl/predict_zones"
        else
            "$baseUrl/predict_points"
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
        println("response.body ${response.body<SegmentationPrediction>()}")
        return response.body()
    }

    /**
     * Releases owned HTTP client resources.
     */
    fun close() {
        httpClient.close()
    }
}

private const val PYTHON_SEGMENTATION_SERVER_URL = "http://127.0.0.1:8001"