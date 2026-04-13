package anton.axenov

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.utils.io.core.Closeable

/**
 * Client for talking with segmentation server.
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
     * Requests segmentation server health.
     *
     * @return health response.
     */
    suspend fun requestHealth(): ServerHealthResponse {
        return httpClient
            .get("$normalizedBaseUrl/health")
            .body()
    }

    /**
     * Uploads one stored zone snapshot.
     *
     * @param payload serializable snapshot payload.
     * @return upload response.
     */
    suspend fun uploadSnapshot(payload: ZoneSnapshotUploadDto): SnapshotUploadResponse {
        return httpClient
            .post("$normalizedBaseUrl/snapshots") {
                contentType(ContentType.Application.Json)
                setBody(payload)
            }
            .body()
    }

    /**
     * Fetches zone statuses from the server.
     *
     * @return current statuses for known zones.
     */
    suspend fun fetchZoneStatuses(): List<ZoneStatus> {
        return httpClient
            .get("$normalizedBaseUrl/zone-statuses")
            .body()
    }

    /**
     * Fetches current zone texts keyed by zone id.
     *
     * @return map of zone id to server text.
     */
    suspend fun fetchZoneTexts(): Map<Long, String> {
        return fetchZoneStatuses()
            .associate { status -> status.zone to status.text }
    }

    /**
     * Releases HTTP client resources.
     */
    override fun close() {
        httpClient.close()
    }
}
