package anton.axenov.localServer

import anton.axenov.AddWorldPointDto
import anton.axenov.DetectionFrameSnapshotDto
import anton.axenov.DeleteRequestResponse
import anton.axenov.SegmentationClient
import anton.axenov.SegmentationPrediction
import anton.axenov.SegmentationPredictionProvider
import anton.axenov.SegmentationSessionProcessor
import anton.axenov.ServerHealthResponse
import anton.axenov.ServerWorldPointDto
import anton.axenov.SnapshotUploadResponse
import anton.axenov.ZoneSnapshotUploadDto
import anton.axenov.ZoneStatus
import anton.axenov.WorldPointMutationResponse
import anton.axenov.generateRequestIdentifier

/**
 * On-device segmentation client backed by the shared session-processing engine.
 *
 * Only raw image prediction is delegated to the supplied provider. Queueing,
 * statuses, triangulation and reconstructed result storage run on the phone.
 *
 * @param predictor current on-device or temporary remote prediction adapter.
 */
class LocalClient(
    predictor: SegmentationPredictionProvider,
    override val sessionId: String = generateRequestIdentifier(),
) : SegmentationClient {
    private val processor = SegmentationSessionProcessor(
        predictor = predictor,
        sessionId = sessionId,
    )

    /**
     * Returns local processor health.
     *
     * @return local processor health response.
     */
    override suspend fun requestHealth(): ServerHealthResponse {
        return processor.requestHealth()
    }

    /**
     * Clears local session state.
     *
     * @return session reset response.
     */
    override suspend fun startNewSession(): ServerHealthResponse {
        return processor.startNewSession()
    }

    /**
     * Queues one snapshot for on-device point processing.
     *
     * @param payload zone snapshot payload.
     * @return queue acceptance response.
     */
    override suspend fun predictPoints(payload: ZoneSnapshotUploadDto): SnapshotUploadResponse {
        return processor.predictPoints(payload)
    }

    /**
     * Deletes one queued local processing request.
     *
     * @param requestId logical request identifier to remove.
     * @return deletion result.
     */
    override suspend fun deleteRequest(requestId: String): DeleteRequestResponse {
        return processor.deleteRequest(requestId)
    }

    /**
     * Detects zones through the current local prediction provider.
     *
     * @param frameSnapshot frame to predict.
     * @return segmentation prediction.
     */
    override suspend fun predictZones(frameSnapshot: DetectionFrameSnapshotDto): SegmentationPrediction {
        return processor.predictZones(frameSnapshot)
    }

    /**
     * Returns local zone processing statuses.
     *
     * @return zone statuses.
     */
    override suspend fun fetchZoneStatuses(): List<ZoneStatus> {
        return processor.fetchZoneStatuses()
    }

    /**
     * Returns points reconstructed on the phone.
     *
     * @return reconstructed world points.
     */
    override suspend fun fetchWorldPoints(): List<ServerWorldPointDto> {
        return processor.fetchWorldPoints()
    }

    /**
     * Adds one manually specified local world point.
     *
     * @param request point position, confidence and optional zone.
     * @return local mutation result.
     */
    override suspend fun addWorldPoint(request: AddWorldPointDto): WorldPointMutationResponse {
        return processor.addWorldPoint(request)
    }

    /**
     * Deletes one local world point.
     *
     * @param pointId stable point identifier.
     * @return local mutation result.
     */
    override suspend fun deleteWorldPoint(pointId: Long): WorldPointMutationResponse {
        return processor.deleteWorldPoint(pointId)
    }

    /**
     * Rotates one local point assignment between its four nearest zones.
     *
     * @param pointId stable point identifier.
     * @return local mutation result.
     */
    override suspend fun rotateWorldPointZone(pointId: Long): WorldPointMutationResponse {
        return processor.rotateWorldPointZone(pointId)
    }

    /**
     * Releases local processing resources.
     */
    override fun close() {
        processor.close()
    }
}
