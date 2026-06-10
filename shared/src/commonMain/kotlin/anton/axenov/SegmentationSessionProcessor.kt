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
    private val workerCount: Int = DEFAULT_SEGMENTATION_WORKER_COUNT,
    private val onSnapshotAccepted: (filename: String, payload: ZoneSnapshotUploadDto) -> Unit = { _, _ -> },
    private val timeSource: TimeSource.WithComparableMarks = TimeSource.Monotonic,
) : SegmentationClient {
    private val stateMutex = Mutex()
    private val snapshotsByZoneId = mutableMapOf<Long, MutableList<StoredSnapshotRecord>>()
    private val triangulationManagersByZoneId = mutableMapOf<Long, ZoneTriangulationManager>()
    private var queue = Channel<QueuedSegmentationTask>(Channel.UNLIMITED)
    private var scope = newProcessorScope()
    private val workerJobs = mutableListOf<Job>()
    private var assignedWorldPointsCache = emptyList<AssignedWorldPoint>()
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
        val filename = "${payload.frameSnapshot.frameTimestamp}-zone-${payload.zone.id}.png"
        onSnapshotAccepted(filename, payload)
        val snapshotCount = stateMutex.withLock {
            val zoneSnapshots = snapshotsByZoneId.getOrPut(payload.zone.id) { mutableListOf() }
            triangulationManagersByZoneId.getOrPut(payload.zone.id) { ZoneTriangulationManager() }
            val record = StoredSnapshotRecord(payload)
            zoneSnapshots += record
            queue.trySend(QueuedSegmentationTask(payload.zone.id, record))
            zoneSnapshots.size
        }
        return SnapshotUploadResponse(
            ok = true,
            zoneId = payload.zone.id,
            snapshotCount = snapshotCount,
            message = "stored snapshot for zone ${payload.zone.id} and queued segmentation",
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
            assignedWorldPointsCache.map { assignedWorldPoint ->
                ServerWorldPointDto(
                    zoneId = assignedWorldPoint.zoneId,
                    position = assignedWorldPoint.worldPoint.position,
                    confidence = assignedWorldPoint.worldPoint.confidence,
                )
            }
        }
    }

    override fun close() {
        workerJobs.forEach { worker -> worker.cancel() }
        workerJobs.clear()
        queue.close()
        scope.cancel()
        predictor.close()
    }

    /**
     * Clears session data and drops all queued tasks.
     */
    suspend fun clear() {
        stateMutex.withLock {
            snapshotsByZoneId.clear()
            triangulationManagersByZoneId.clear()
            assignedWorldPointsCache = emptyList()
            hasPendingZoneSeparation = false
            hasZoneSeparationRun = false
            lastZoneSeparationAt = timeSource.markNow()
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
                    val task = queue.receiveCatching().getOrNull() ?: break
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
        stateMutex.withLock {
            task.record.segmentationState = SegmentationState.PROCESSING
            task.record.segmentationError = null
        }
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
        queue.close()
        scope.cancel()
        queue = Channel<QueuedSegmentationTask>(Channel.UNLIMITED)
        scope = newProcessorScope()
        startWorkers()
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

        val zones = snapshotsByZoneId.values
            .mapNotNull { snapshots -> snapshots.firstOrNull()?.snapshot?.zone }
            .distinctBy { zone -> zone.id }
            .sortedBy { zone -> zone.id }
        val worldPoints = triangulationManagersByZoneId.values
            .flatMap { manager -> manager.getResolvedWorldPoints() }
        assignedWorldPointsCache = assignWorldPointsToZones(worldPoints, zones)
        hasPendingZoneSeparation = false
        hasZoneSeparationRun = true
        lastZoneSeparationAt = timeSource.markNow()
    }
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

private const val DEFAULT_SEGMENTATION_WORKER_COUNT = 3
private val ZONE_SEPARATION_INTERVAL = 10.seconds
