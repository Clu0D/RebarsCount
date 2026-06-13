package anton.axenov

import korlibs.math.geom.Vector3F
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

/**
 * Triangulated world-space point derived from corresponding 2D segmentation points.
 *
 * @param position reconstructed 3D point in world coordinates.
 * @param parentPoints source 2D points used to reconstruct this world point.
 * @param confidence reconstruction confidence in range `[0, 1]`.
 */
data class WorldPoint(
    val position: Vector3F,
    val parentPoints: Set<ZoneTriangulationPoint>,
    val confidence: Float,
)

/**
 * Shared in-memory processing session used by both Ktor server and Android local client.
 *
 * @param predictor provider used to obtain Python or local model predictions.
 * @param workerCount number of parallel point-prediction workers.
 * @param onSnapshotAccepted optional platform-specific snapshot debug callback.
 */
class SegmentationSessionProcessor(
    private val predictor: SegmentationPredictionProvider,
    override val sessionId: String = "",
    private val workerCount: Int = DEFAULT_SEGMENTATION_WORKER_COUNT,
    private val onSnapshotAccepted: (filename: String, payload: ZoneSnapshotUploadDto) -> Unit = { _, _ -> },
    private val timeSource: TimeSource.WithComparableMarks = TimeSource.Monotonic,
    private val closePredictorOnClose: Boolean = true,
) : SegmentationClient {
    private val stateMutex = Mutex()
    private val snapshotsByZoneId = mutableMapOf<Long, MutableList<StoredSnapshotRecord>>()
    private val triangulationManagersByZoneId = mutableMapOf<Long, ZoneTriangulationManager>()
    private val queuedTasks = mutableListOf<QueuedSegmentationTask>()
    private var queueSignals = Channel<Unit>(Channel.UNLIMITED)
    private var scope = newProcessorScope()
    private val workerJobs = mutableListOf<Job>()
    private var assignedWorldPointsCache = emptyList<IdentifiedAssignedWorldPoint>()
    private val pointIdByAutomaticWorldPoint = mutableMapOf<WorldPoint, Long>()
    private val manuallyAddedWorldPointsById = mutableMapOf<Long, IdentifiedAssignedWorldPoint>()
    private val deletedAutomaticWorldPointIds = mutableSetOf<Long>()
    private val overriddenZoneIdByPointId = mutableMapOf<Long, Long>()
    private var nextWorldPointId = 1L
    private var hasPendingZoneSeparation = false
    private var hasZoneSeparationRun = false
    private var lastZoneSeparationAt = timeSource.markNow()

    init {
        startWorkers()
    }

    override suspend fun requestHealth(): ServerHealthResponse {
        return ServerHealthResponse(ok = true, message = "Segmentation processor is online")
    }

    override suspend fun startNewSession(): ServerHealthResponse {
        clear()
        return ServerHealthResponse(
            ok = true,
            message = "Started new session and cleared processing state",
        )
    }

    override suspend fun predictPoints(payload: ZoneSnapshotUploadDto): SnapshotUploadResponse {
        requireSessionMatch(payload.sessionId)
        val filename = "${payload.frameSnapshot.frameTimestamp}-zone-${payload.zone.id}.png"
        onSnapshotAccepted(filename, payload)
        val snapshotCount = stateMutex.withLock {
            val zoneSnapshots = snapshotsByZoneId.getOrPut(payload.zone.id) { mutableListOf() }
            triangulationManagersByZoneId.getOrPut(payload.zone.id) { ZoneTriangulationManager() }
            val record = StoredSnapshotRecord(payload)
            zoneSnapshots += record
            queuedTasks += QueuedSegmentationTask(payload.zone.id, record)
            zoneSnapshots.size
        }
        if (queueSignals.trySend(Unit).isFailure) {
            stateMutex.withLock {
                snapshotsByZoneId[payload.zone.id]?.removeAll { record ->
                    record.snapshot.requestId == payload.requestId &&
                        record.segmentationState == SegmentationState.QUEUED
                }
                queuedTasks.removeAll { task ->
                    task.record.snapshot.requestId == payload.requestId &&
                        task.record.segmentationState == SegmentationState.QUEUED
                }
            }
            error("Segmentation queue is unavailable for session $sessionId")
        }
        return SnapshotUploadResponse(
            ok = true,
            zoneId = payload.zone.id,
            requestId = payload.requestId,
            snapshotCount = snapshotCount,
            message = "stored snapshot for zone ${payload.zone.id} and queued segmentation",
        )
    }

    override suspend fun deleteRequest(requestId: String): DeleteRequestResponse {
        val removedSnapshots = stateMutex.withLock {
            var removedSnapshots = 0
            snapshotsByZoneId.values.forEach { snapshots ->
                val beforeCount = snapshots.size
                snapshots.removeAll { record ->
                    record.snapshot.requestId == requestId &&
                        record.segmentationState == SegmentationState.QUEUED
                }
                if (snapshots.size < beforeCount) {
                    removedSnapshots += beforeCount - snapshots.size
                }
            }
            queuedTasks.removeAll { task ->
                task.record.snapshot.requestId == requestId &&
                    task.record.segmentationState == SegmentationState.QUEUED
            }
            removedSnapshots
        }
        return DeleteRequestResponse(
            ok = true,
            requestId = requestId,
            removedSnapshots = removedSnapshots,
            message = if (removedSnapshots > 0) {
                "Removed $removedSnapshots queued snapshot(s)"
            } else {
                "No queued snapshot matched request $requestId in session $sessionId"
            },
        )
    }

    override suspend fun predictZones(frameSnapshot: DetectionFrameSnapshotDto): SegmentationPrediction {
        return predictor.predict(
            imageBytes = frameSnapshot.screenshotPngBytes,
            filename = "${frameSnapshot.frameTimestamp}-zones-seg.png",
            zonePrediction = true,
        )
    }

    override suspend fun fetchZoneStatuses(): List<ZoneStatus> {
        return stateMutex.withLock {
            snapshotsByZoneId.keys.sorted().map { zoneId ->
                val snapshots = snapshotsByZoneId[zoneId].orEmpty()
                val queued = snapshots.count { it.segmentationState == SegmentationState.QUEUED }
                val processing = snapshots.count { it.segmentationState == SegmentationState.PROCESSING }
                val completed = snapshots.count { it.segmentationState == SegmentationState.COMPLETED }
                val failed = snapshots.count { it.segmentationState == SegmentationState.FAILED }
                ZoneStatus(
                    zone = zoneId,
                    text = "${snapshots.size} snapshots, queued=$queued, processing=$processing, " +
                        "completed=$completed, failed=$failed",
                    total = snapshots.size,
                    queued = queued,
                    processing = processing,
                    completed = completed,
                    failed = failed,
                )
            }
        }
    }

    override suspend fun fetchWorldPoints(): List<ServerWorldPointDto> {
        return stateMutex.withLock {
            refreshAssignedWorldPointsIfNeeded()
            assignedWorldPointsCache.map { assignedWorldPoint -> assignedWorldPoint.toDto() }
        }
    }

    override suspend fun addWorldPoint(request: AddWorldPointDto): WorldPointMutationResponse {
        return stateMutex.withLock {
            refreshAssignedWorldPointsIfNeeded()
            if (!request.position.isFinite()) {
                return@withLock WorldPointMutationResponse(
                    ok = false,
                    message = "World point position must contain only finite coordinates",
                )
            }
            if (!request.confidence.isFinite() || request.confidence !in 0f..1f) {
                return@withLock WorldPointMutationResponse(
                    ok = false,
                    message = "World point confidence must be in range [0, 1]",
                )
            }
            val zones = knownZones()
            if (zones.isEmpty()) {
                return@withLock WorldPointMutationResponse(
                    ok = false,
                    message = "Cannot add a world point before at least one zone is known",
                )
            }
            val assignedZoneId = request.zoneId?.let { requestedZoneId ->
                if (zones.none { zone -> zone.id == requestedZoneId }) {
                    return@withLock WorldPointMutationResponse(
                        ok = false,
                        message = "Zone $requestedZoneId is not known in session $sessionId",
                    )
                }
                requestedZoneId
            } ?: nearestZoneId(request.position, zones)
            val identifiedPoint = IdentifiedAssignedWorldPoint(
                pointId = nextWorldPointId++,
                assignedWorldPoint = AssignedWorldPoint(
                    zoneId = assignedZoneId,
                    worldPoint = WorldPoint(
                        position = request.position,
                        parentPoints = emptySet(),
                        confidence = request.confidence,
                    ),
                    isAnchor = false,
                ),
            )
            manuallyAddedWorldPointsById[identifiedPoint.pointId] = identifiedPoint
            assignedWorldPointsCache = assignedWorldPointsCache + identifiedPoint
            WorldPointMutationResponse(
                ok = true,
                point = identifiedPoint.toDto(),
                message = "Added world point ${identifiedPoint.pointId} to zone $assignedZoneId",
            )
        }
    }

    override suspend fun deleteWorldPoint(pointId: Long): WorldPointMutationResponse {
        return stateMutex.withLock {
            refreshAssignedWorldPointsIfNeeded()
            val point = assignedWorldPointsCache.firstOrNull { candidate -> candidate.pointId == pointId }
                ?: return@withLock WorldPointMutationResponse(
                    ok = false,
                    message = "World point $pointId was not found in session $sessionId",
                )
            if (manuallyAddedWorldPointsById.remove(pointId) == null) {
                deletedAutomaticWorldPointIds += pointId
            }
            overriddenZoneIdByPointId.remove(pointId)
            assignedWorldPointsCache = assignedWorldPointsCache.filterNot { candidate ->
                candidate.pointId == point.pointId
            }
            WorldPointMutationResponse(
                ok = true,
                message = "Deleted world point $pointId",
            )
        }
    }

    override suspend fun rotateWorldPointZone(pointId: Long): WorldPointMutationResponse {
        return stateMutex.withLock {
            refreshAssignedWorldPointsIfNeeded()
            val pointIndex = assignedWorldPointsCache.indexOfFirst { candidate -> candidate.pointId == pointId }
            if (pointIndex < 0) {
                return@withLock WorldPointMutationResponse(
                    ok = false,
                    message = "World point $pointId was not found in session $sessionId",
                )
            }
            val point = assignedWorldPointsCache[pointIndex]
            val nearestZones = knownZones()
                .sortedWith(
                    compareBy<Zone> { zone -> squaredDistance(point.worldPoint.position, zone.center) }
                        .thenBy { zone -> zone.id },
                )
                .take(MAX_ROTATING_ZONE_COUNT)
            if (nearestZones.isEmpty()) {
                return@withLock WorldPointMutationResponse(
                    ok = false,
                    point = point.toDto(),
                    message = "Cannot rotate world point $pointId because no zones are known",
                )
            }
            val currentZoneIndex = nearestZones.indexOfFirst { zone -> zone.id == point.zoneId }
            val nextZone = if (currentZoneIndex < 0) {
                nearestZones.first()
            } else {
                nearestZones[(currentZoneIndex + 1) % nearestZones.size]
            }
            val updatedPoint = point.withZoneId(nextZone.id)
            overriddenZoneIdByPointId[pointId] = nextZone.id
            if (manuallyAddedWorldPointsById.containsKey(pointId)) {
                manuallyAddedWorldPointsById[pointId] = updatedPoint
            }
            assignedWorldPointsCache = assignedWorldPointsCache.toMutableList().also { points ->
                points[pointIndex] = updatedPoint
            }
            WorldPointMutationResponse(
                ok = true,
                point = updatedPoint.toDto(),
                message = "Moved world point $pointId to zone ${nextZone.id}",
            )
        }
    }

    override fun close() {
        workerJobs.forEach { worker -> worker.cancel() }
        workerJobs.clear()
        queueSignals.close()
        scope.cancel()
        if (closePredictorOnClose) {
            predictor.close()
        }
    }

    /**
     * Clears session data and drops all queued tasks.
     */
    suspend fun clear() {
        stateMutex.withLock {
            snapshotsByZoneId.clear()
            triangulationManagersByZoneId.clear()
            assignedWorldPointsCache = emptyList()
            pointIdByAutomaticWorldPoint.clear()
            manuallyAddedWorldPointsById.clear()
            deletedAutomaticWorldPointIds.clear()
            overriddenZoneIdByPointId.clear()
            nextWorldPointId = 1L
            hasPendingZoneSeparation = false
            hasZoneSeparationRun = false
            lastZoneSeparationAt = timeSource.markNow()
            queuedTasks.clear()
            restartWorkers()
        }
    }

    /**
     * Starts point-prediction workers for the current queue.
     */
    private fun startWorkers() {
        repeat(workerCount) {
            workerJobs += scope.launch(Dispatchers.Default) {
                while (true) {
                    queueSignals.receiveCatching().getOrNull() ?: break
                    val task = stateMutex.withLock {
                        val queuedTask = queuedTasks.removeFirstOrNullCompat() ?: return@withLock null
                        queuedTask.record.segmentationState = SegmentationState.PROCESSING
                        queuedTask.record.segmentationError = null
                        queuedTask
                    } ?: continue
                    processTask(task)
                }
            }
        }
    }

    /**
     * Processes one queued point-recognition task and updates shared session state.
     *
     * @param task queued snapshot task.
     */
    private suspend fun processTask(task: QueuedSegmentationTask) {
        val filename = "${task.record.snapshot.frameSnapshot.frameTimestamp}-zone-${task.zoneId}.png"
        runCatching {
            predictor.predict(
                imageBytes = task.record.snapshot.frameSnapshot.screenshotPngBytes,
                filename = filename,
                zonePrediction = false,
            )
        }.onSuccess { prediction ->
            stateMutex.withLock {
                task.record.segmentationState = SegmentationState.COMPLETED
                triangulationManagersByZoneId[task.zoneId]?.addSegmentationResult(
                    zone = task.record.snapshot.zone,
                    frameSnapshot = task.record.snapshot.frameSnapshot,
                    prediction = prediction,
                )
                hasPendingZoneSeparation = true
            }
        }.onFailure { error ->
            stateMutex.withLock {
                task.record.segmentationState = SegmentationState.FAILED
                task.record.segmentationError = error.message ?: "Unknown prediction error"
            }
        }
    }

    /**
     * Recreates workers and queue after a session reset.
     */
    private fun restartWorkers() {
        workerJobs.forEach { worker -> worker.cancel() }
        workerJobs.clear()
        queueSignals.close()
        scope.cancel()
        queueSignals = Channel(Channel.UNLIMITED)
        scope = newProcessorScope()
        queuedTasks.clear()
        startWorkers()
    }

    /**
     * Verifies that incoming payloads belong to this processor session.
     *
     * @param candidateSessionId session identifier attached to the request.
     */
    private fun requireSessionMatch(candidateSessionId: String) {
        if (sessionId.isBlank() || candidateSessionId.isBlank()) {
            return
        }
        require(candidateSessionId == sessionId) {
            "Request session $candidateSessionId does not match processor session $sessionId"
        }
    }

    /**
     * Rebuilds cached zone assignments only when new frame data appeared and the throttle allows it.
     */
    private fun refreshAssignedWorldPointsIfNeeded() {
        val shouldRecompute = when {
            !hasZoneSeparationRun -> true
            !hasPendingZoneSeparation -> false
            assignedWorldPointsCache.isEmpty() -> true
            lastZoneSeparationAt.elapsedNow() >= ZONE_SEPARATION_INTERVAL -> true
            else -> false
        }
        if (!shouldRecompute) {
            return
        }

        val zones = knownZones()
        val worldPoints = triangulationManagersByZoneId.values
            .flatMap { manager -> manager.getResolvedWorldPoints() }
        val automaticPoints = assignWorldPointsToZones(worldPoints, zones)
            .map { assignedPoint ->
                val pointId = pointIdByAutomaticWorldPoint.getOrPut(assignedPoint.worldPoint) {
                    nextWorldPointId++
                }
                IdentifiedAssignedWorldPoint(
                    pointId = pointId,
                    assignedWorldPoint = overriddenZoneIdByPointId[pointId]
                        ?.let { zoneId -> assignedPoint.withZoneId(zoneId) }
                        ?: assignedPoint,
                )
            }
            .filterNot { identifiedPoint -> identifiedPoint.pointId in deletedAutomaticWorldPointIds }
        assignedWorldPointsCache = automaticPoints + manuallyAddedWorldPointsById.values
        hasPendingZoneSeparation = false
        hasZoneSeparationRun = true
        lastZoneSeparationAt = timeSource.markNow()
    }

    /**
     * Returns all known zones in deterministic identifier order.
     *
     * @return zones registered by uploaded snapshots.
     */
    private fun knownZones(): List<Zone> {
        return snapshotsByZoneId.values
            .mapNotNull { snapshots -> snapshots.firstOrNull()?.snapshot?.zone }
            .distinctBy { zone -> zone.id }
            .sortedBy { zone -> zone.id }
    }
}

/**
 * One assigned world point paired with its stable post-processing identifier.
 *
 * @param pointId stable identifier exposed to clients.
 * @param assignedWorldPoint current point assignment.
 */
private data class IdentifiedAssignedWorldPoint(
    val pointId: Long,
    val assignedWorldPoint: AssignedWorldPoint,
) {
    val zoneId: Long
        get() = assignedWorldPoint.zoneId

    val worldPoint: WorldPoint
        get() = assignedWorldPoint.worldPoint

    /**
     * Returns this identified point assigned to another zone.
     *
     * @param zoneId new zone identifier.
     * @return copied identified point.
     */
    fun withZoneId(zoneId: Long): IdentifiedAssignedWorldPoint {
        return copy(assignedWorldPoint = assignedWorldPoint.withZoneId(zoneId))
    }

    /**
     * Converts this point to the public API representation.
     *
     * @return serializable world-point DTO.
     */
    fun toDto(): ServerWorldPointDto {
        return ServerWorldPointDto(
            pointId = pointId,
            zoneId = zoneId,
            position = worldPoint.position,
            confidence = worldPoint.confidence,
        )
    }
}

/**
 * Returns this assignment with a changed zone identifier.
 *
 * @param zoneId new zone identifier.
 * @return copied assignment.
 */
private fun AssignedWorldPoint.withZoneId(zoneId: Long): AssignedWorldPoint {
    return copy(zoneId = zoneId, isAnchor = false)
}

/**
 * Checks that all vector coordinates are finite.
 *
 * @return true when every coordinate is finite.
 */
private fun Vector3F.isFinite(): Boolean {
    return x.isFinite() && y.isFinite() && z.isFinite()
}

/**
 * Returns the nearest zone identifier for one world-space position.
 *
 * @param point world-space position.
 * @param zones candidate zones.
 * @return identifier of the nearest zone.
 */
private fun nearestZoneId(point: Vector3F, zones: List<Zone>): Long {
    return zones.minWith(
        compareBy<Zone> { zone -> squaredDistance(point, zone.center) }
            .thenBy { zone -> zone.id },
    ).id
}

/**
 * Computes squared Euclidean distance between two world positions.
 *
 * @param first first position.
 * @param second second position.
 * @return squared distance.
 */
private fun squaredDistance(first: Vector3F, second: Vector3F): Float {
    val delta = first - second
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
}

private data class StoredSnapshotRecord(
    val snapshot: ZoneSnapshotUploadDto,
    var segmentationState: SegmentationState = SegmentationState.QUEUED,
    var segmentationError: String? = null,
)

private data class QueuedSegmentationTask(
    val zoneId: Long,
    val record: StoredSnapshotRecord,
)

/**
 * Creates an isolated scope for shared processing workers.
 *
 * @return processor coroutine scope.
 */
private fun newProcessorScope(): CoroutineScope {
    return CoroutineScope(SupervisorJob() + Dispatchers.Default)
}

/**
 * Removes and returns the first list element.
 *
 * @return first element or null when the list is empty.
 */
private fun <T> MutableList<T>.removeFirstOrNullCompat(): T? {
    if (isEmpty()) {
        return null
    }
    return removeAt(0)
}

private const val DEFAULT_SEGMENTATION_WORKER_COUNT = 3
private const val MAX_ROTATING_ZONE_COUNT = 4
private val ZONE_SEPARATION_INTERVAL = 10.seconds
