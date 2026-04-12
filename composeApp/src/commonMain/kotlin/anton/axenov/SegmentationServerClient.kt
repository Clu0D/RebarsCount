package anton.axenov

import io.ktor.client.HttpClient
import io.ktor.client.request.get
import io.ktor.client.statement.bodyAsText
import io.ktor.utils.io.core.Closeable

/**
 * Requests health information from the segmentation server.
 *
 * @param baseUrl server base URL without endpoint path.
 * @param httpClient HTTP client used to perform requests.
 */
class SegmentationServerClient(
    baseUrl: String,
    private val httpClient: HttpClient,
) : Closeable {
    private val normalizedBaseUrl = baseUrl.trimEnd('/')

    /**
     * Requests the segmentation server health endpoint.
     *
     * @return response body returned by the server.
     */
    suspend fun requestHealth(): String =
        httpClient.get("$normalizedBaseUrl/health").bodyAsText()

    /**
     * Releases HTTP client resources.
     */
    override fun close() {
        httpClient.close()
    }
}
