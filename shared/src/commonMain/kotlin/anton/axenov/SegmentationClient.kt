package anton.axenov

/**
 * Common segmentation client contract.
 */
interface SegmentationClient {
    /**
     * Stable identifier of the current processing session.
     */
    val sessionId: String

    /**
     * Returns processing service health.
     *
     * @return current health response.
     */
    suspend fun requestHealth(): ServerHealthResponse

    /**
     * Starts a new processing session and clears previous in-memory state.
     *
     * @return session reset response.
     */
    suspend fun startNewSession(): ServerHealthResponse

    /**
     * Queues one zone snapshot for point recognition.
     *
     * @param payload zone snapshot to process.
     * @return queue acceptance response.
     */
    suspend fun predictPoints(payload: ZoneSnapshotUploadDto): SnapshotUploadResponse

    /**
     * Deletes one queued processing request from the current session.
     *
     * @param requestId logical request identifier to remove.
     * @return deletion result.
     */
    suspend fun deleteRequest(requestId: String): DeleteRequestResponse

    /**
     * Detects zones in one frame.
     *
     * @param frameSnapshot frame to process.
     * @return detected segmentation instances.
     */
    suspend fun predictZones(frameSnapshot: DetectionFrameSnapshotDto): SegmentationPrediction

    /**
     * Returns current processing statuses for known zones.
     *
     * @return zone statuses.
     */
    suspend fun fetchZoneStatuses(): List<ZoneStatus>

    /**
     * Returns all currently reconstructed world points.
     *
     * @return reconstructed points.
     */
    suspend fun fetchWorldPoints(): List<ServerWorldPointDto>

    /**
     * Adds one manually specified world point to post-processed results.
     *
     * @param request point position, confidence and optional explicit zone.
     * @return mutation result containing the created point.
     */
    suspend fun addWorldPoint(request: AddWorldPointDto): WorldPointMutationResponse

    /**
     * Deletes one world point from post-processed results.
     *
     * @param pointId stable identifier of the point to delete.
     * @return deletion result.
     */
    suspend fun deleteWorldPoint(pointId: Long): WorldPointMutationResponse

    /**
     * Moves one world point to the next zone among its four nearest zone centers.
     *
     * @param pointId stable identifier of the point whose zone should rotate.
     * @return mutation result containing the updated point.
     */
    suspend fun rotateWorldPointZone(pointId: Long): WorldPointMutationResponse

    /**
     * Returns current zone status texts keyed by zone identifier.
     *
     * @return zone status text map.
     */
    suspend fun fetchZoneTexts(): Map<Long, String> {
        return fetchZoneStatuses().associate { status -> status.zone to status.text }
    }

    /**
     * Releases resources owned by this client.
     */
    fun close()
}

/**
 * Common prediction adapter implemented by server-side and Android gateways.
 */
interface SegmentationPredictionProvider {
    /**
     * Predicts segmentation for one PNG image.
     *
     * @param imageBytes PNG image bytes.
     * @param filename logical image filename.
     * @param zonePrediction true for zone detection and false for point detection.
     * @return segmentation prediction.
     */
    suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
        zonePrediction: Boolean,
    ): SegmentationPrediction

    /**
     * Releases provider resources.
     */
    fun close()
}
