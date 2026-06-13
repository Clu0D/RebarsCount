package anton.axenov

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.HttpRequestBuilder
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.accept
import io.ktor.client.request.header
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess

/**
 * Client for talking with segmentation server.
 *
 * @param baseUrl server base URL without endpoint path.
 * @param httpClient HTTP client used to perform requests.
 */
class SegmentationServerClient(
    baseUrl: String,
    private val httpClient: HttpClient,
    override val sessionId: String = generateRequestIdentifier(),
) : SegmentationClient {
    private val normalizedBaseUrl = baseUrl.trimEnd('/')

    /**
     * Requests segmentation server health.
     *
     * @return health response.
     */
    override suspend fun requestHealth(): ServerHealthResponse {
        return httpClient
            .get("$normalizedBaseUrl/health") {
                applyRequestMetadata()
            }
            .bodyOrThrow("Health request")
    }

    /**
     * Starts a new server-side session and clears old in-memory state.
     *
     * @return server response confirming session reset.
     */
    override suspend fun startNewSession(): ServerHealthResponse {
        return httpClient
            .post("$normalizedBaseUrl/start_new_session") {
                applyRequestMetadata()
                contentType(ContentType.Application.Json)
            }
            .bodyOrThrow("Starting a new session")
    }

    /**
     * Requests points prediction.
     *
     * @param payload serializable snapshot payload.
     * @return upload response.
     */
    override suspend fun predictPoints(payload: ZoneSnapshotUploadDto): SnapshotUploadResponse {
        return httpClient
            .post("$normalizedBaseUrl/predict_points") {
                applyRequestMetadata()
                accept(ContentType.Application.Json)
                contentType(ContentType.Application.Json)
                setBody(payload)
            }
            .bodyOrThrow("Point prediction")
    }

    /**
     * Deletes one queued snapshot-processing request from the current session.
     *
     * @param requestId logical request identifier to remove.
     * @return deletion result.
     */
    override suspend fun deleteRequest(requestId: String): DeleteRequestResponse {
        return httpClient
            .post("$normalizedBaseUrl/delete_request") {
                applyRequestMetadata()
                accept(ContentType.Application.Json)
                contentType(ContentType.Application.Json)
                setBody(DeleteRequestDto(requestId = requestId))
            }
            .bodyOrThrow("Deleting request $requestId")
    }

    /**
     * Requests zones prediction.
     *
     * @param frameSnapshot screenshot payload to analyze.
     * @return detected interest zones.
     */
    override suspend fun predictZones(frameSnapshot: DetectionFrameSnapshotDto): SegmentationPrediction {
        return httpClient
            .post("$normalizedBaseUrl/predict_zones") {
                applyRequestMetadata()
                accept(ContentType.Application.Json)
                contentType(ContentType.Application.Json)
                setBody(frameSnapshot)
            }
            .bodyOrThrow("Zone prediction")
    }

    /**
     * Fetches zone statuses from the server.
     *
     * @return current statuses for known zones.
     */
    override suspend fun fetchZoneStatuses(): List<ZoneStatus> {
        return httpClient
            .get("$normalizedBaseUrl/zone-statuses") {
                applyRequestMetadata()
            }
            .bodyOrThrow("Fetching zone statuses")
    }

    /**
     * Fetches all reconstructed world points.
     *
     * @return world points currently known by the server.
     */
    override suspend fun fetchWorldPoints(): List<ServerWorldPointDto> {
        return httpClient
            .get("$normalizedBaseUrl/world-points") {
                applyRequestMetadata()
            }
            .bodyOrThrow("Fetching world points")
    }

    /**
     * Adds one manually specified world point.
     *
     * @param request point position, confidence and optional zone.
     * @return server mutation result.
     */
    override suspend fun addWorldPoint(request: AddWorldPointDto): WorldPointMutationResponse {
        return httpClient
            .post("$normalizedBaseUrl/world-points/add") {
                applyRequestMetadata()
                accept(ContentType.Application.Json)
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            .bodyOrThrow("Adding a world point")
    }

    /**
     * Deletes one world point.
     *
     * @param pointId stable point identifier.
     * @return server mutation result.
     */
    override suspend fun deleteWorldPoint(pointId: Long): WorldPointMutationResponse {
        return httpClient
            .post("$normalizedBaseUrl/world-points/delete") {
                applyRequestMetadata()
                accept(ContentType.Application.Json)
                contentType(ContentType.Application.Json)
                setBody(WorldPointIdDto(pointId))
            }
            .bodyOrThrow("Deleting world point $pointId")
    }

    /**
     * Rotates one point assignment between its four nearest zones.
     *
     * @param pointId stable point identifier.
     * @return server mutation result.
     */
    override suspend fun rotateWorldPointZone(pointId: Long): WorldPointMutationResponse {
        return httpClient
            .post("$normalizedBaseUrl/world-points/rotate-zone") {
                applyRequestMetadata()
                accept(ContentType.Application.Json)
                contentType(ContentType.Application.Json)
                setBody(WorldPointIdDto(pointId))
            }
            .bodyOrThrow("Rotating zone for world point $pointId")
    }

    /**
     * Deletes one zone and all related server-side state.
     *
     * @param zoneId stable zone identifier.
     * @return cascading deletion result.
     */
    override suspend fun deleteZone(zoneId: Long): DeleteZoneResponse {
        return httpClient
            .post("$normalizedBaseUrl/zones/delete") {
                applyRequestMetadata()
                accept(ContentType.Application.Json)
                contentType(ContentType.Application.Json)
                setBody(ZoneIdDto(zoneId))
            }
            .bodyOrThrow("Deleting zone $zoneId")
    }

    /**
     * Fetches current zone texts keyed by zone id.
     *
     * @return map of zone id to server text.
     */
    override suspend fun fetchZoneTexts(): Map<Long, String> {
        return fetchZoneStatuses()
            .associate { status -> status.zone to status.text }
    }

    /**
     * Releases HTTP client resources.
     */
    override fun close() {
        httpClient.close()
    }

    /**
     * Adds session correlation headers to one HTTP request.
     */
    private fun HttpRequestBuilder.applyRequestMetadata() {
        header(SESSION_ID_HTTP_HEADER, sessionId)
    }
}

/**
 * Decodes a successful response or throws an error containing the server response body.
 *
 * @param operation user-readable operation name included in an error.
 * @return decoded successful response body.
 */
private suspend inline fun <reified T> HttpResponse.bodyOrThrow(operation: String): T {
    if (!status.isSuccess()) {
        val errorBody = bodyAsText().take(MAX_ERROR_BODY_LENGTH)
        throw IllegalStateException(
            "$operation failed: HTTP ${status.value} ${status.description}: $errorBody",
        )
    }
    return body()
}

private const val MAX_ERROR_BODY_LENGTH = 2_000
