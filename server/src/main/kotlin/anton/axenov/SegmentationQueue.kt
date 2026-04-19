package anton.axenov

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import korlibs.math.geom.Vector3F

/**
 * Triangulated world-space point derived from corresponding 2D segmentation points.
 *
 * @param position reconstructed 3D point in world coordinates.
 * @param parentPoints source 2D points that were used to reconstruct this world point.
 * @param confidence reconstruction confidence in range `[0, 1]`.
 */
data class WorldPoint(
    val position: Vector3F,
    val parentPoints: Set<ZoneTriangulationPoint>,
    val confidence: Float,
)

/**
 * In-memory server state and background queue for Python segmentation.
 *
 * @param workerCount number of parallel Python prediction workers.
 */
class SegmentationQueue(
    private val workerCount: Int = DEFAULT_SEGMENTATION_WORKER_COUNT,
    private val snapshotDebugStore: SnapshotDebugStore = SnapshotDebugStore(),
) {
    private val snapshotsByZoneId = ConcurrentHashMap<Long, CopyOnWriteArrayList<StoredSnapshotRecord>>()
    private val zoneTriangulationManagersByZoneId = ConcurrentHashMap<Long, ZoneTriangulationManager>()
    private var predictor = SegmentationPredictor()
    private var queue = Channel<QueuedSegmentationTask>(Channel.UNLIMITED)
    private var scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val workerJobs = mutableListOf<Job>()

    /**
     * Configures predictor instance and restarts workers.
     *
     * @param predictor predictor used by worker queue.
     */
    fun configure(predictor: SegmentationPredictor) {
        this.predictor.close()
        this.predictor = predictor
        restartWorkers()
    }

    /**
     * Clears snapshots and queue state.
     */
    fun clear() {
        snapshotsByZoneId.clear()
        zoneTriangulationManagersByZoneId.clear()
        restartWorkers()
    }

    /**
     * Stops queue workers and releases coroutine scope.
     */
    fun stop() {
        workerJobs.forEach { workerJob ->
            workerJob.cancel()
        }
        workerJobs.clear()
        queue.close()
        scope.cancel()
        predictor.close()
    }

    /**
     * Schedule segmentation.
     *
     * @param payload uploaded snapshot payload.
     * @return stored snapshot count for the zone.
     */
    fun addSnapshot(payload: ZoneSnapshotUploadDto): Int {
        val zoneSnapshots = snapshotsByZoneId.getOrPut(payload.zone.id) { CopyOnWriteArrayList() }
        zoneTriangulationManagersByZoneId.getOrPut(payload.zone.id) { ZoneTriangulationManager() }
        val filename = "${payload.frameSnapshot.frameTimestamp}-zone-${payload.zone.id}.png"
        snapshotDebugStore.savePredictPointsSnapshot(
            sourceImageFilename = filename,
            payload = payload,
        )
        val record = StoredSnapshotRecord(payload)
        zoneSnapshots += record
        queue.trySend(
            QueuedSegmentationTask(
                zoneId = payload.zone.id,
                snapshotRecord = record,
            ),
        )
        return zoneSnapshots.size
    }

    /**
     * Returns status lines for all known zones.
     */
    fun getZoneStatuses(): List<ZoneStatus> {
        return snapshotsByZoneId.keys
            .toSortedSet()
            .map { zoneId ->
                val snapshots = snapshotsByZoneId[zoneId].orEmpty()
                val queued = snapshots.count { it.segmentationState == SegmentationState.QUEUED }
                val processing = snapshots.count { it.segmentationState == SegmentationState.PROCESSING }
                val completed = snapshots.count { it.segmentationState == SegmentationState.COMPLETED }
                val failed = snapshots.count { it.segmentationState == SegmentationState.FAILED }
                ZoneStatus(
                    zone = zoneId,
                    text = "${snapshots.size} snapshots, queued=$queued, processing=$processing, " +
                            "completed=$completed, failed=$failed",
                )
            }
    }

    /**
     * Returns all reconstructed world points from all known zones.
     *
     * @return flattened world points ordered by zone id.
     */
    fun getAllWorldPoints(): List<ServerWorldPointDto> {
        return zoneTriangulationManagersByZoneId
            .toSortedMap()
            .flatMap { (zoneId, manager) ->
                synchronized(manager) {
                    manager.getWorldPoints()
                        .map { worldPoint ->
                            ServerWorldPointDto(
                                zoneId = zoneId,
                                position = worldPoint.position,
                                confidence = worldPoint.confidence,
                            )
                        }
                }
            }
    }

    /**
     * Restarts background queue workers and drops queued.
     */
    private fun restartWorkers() {
        workerJobs.forEach { workerJob ->
            workerJob.cancel()
        }
        workerJobs.clear()
        queue.close()
        scope.cancel()

        queue = Channel<QueuedSegmentationTask>(Channel.UNLIMITED)
        scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

        repeat(workerCount) {
            workerJobs += scope.launch {
                while (true) {
                    val task = queue.receiveCatching().getOrNull() ?: break
                    val record = task.snapshotRecord
                    record.segmentationState = SegmentationState.PROCESSING
                    record.segmentationError = null
                    record.segmentationResult = null

                    val filename = "${record.snapshot.frameSnapshot.frameTimestamp}-zone-${task.zoneId}.png"
                    runCatching {
                        predictor.predict(
                            imageBytes = record.snapshot.frameSnapshot.screenshotPngBytes,
                            filename = filename,
                            zonePrediction = false,
                        )
                    }.onSuccess { prediction ->
                        record.segmentationState = SegmentationState.COMPLETED
                        record.segmentationResult = prediction
                        zoneTriangulationManagersByZoneId[task.zoneId]?.let { zoneTriangulationManager ->
                            synchronized(zoneTriangulationManager) {
                                zoneTriangulationManager.addSegmentationResult(
                                    frameSnapshot = record.snapshot.frameSnapshot,
                                    prediction = prediction,
                                )
                            }
                        }
                    }.onFailure { error ->
                        record.segmentationState = SegmentationState.FAILED
                        record.segmentationError = error.message ?: error.javaClass.simpleName
                    }
                }
            }
        }
    }
}

/**
 * Snapshot segmentation status and result.
 *
 * @param snapshot original uploaded payload.
 */
private data class StoredSnapshotRecord(
    val snapshot: ZoneSnapshotUploadDto,
) {
    @Volatile
    var segmentationState: SegmentationState = SegmentationState.QUEUED

    @Volatile
    var segmentationResult: SegmentationPrediction? = null

    @Volatile
    var segmentationError: String? = null
}

private data class QueuedSegmentationTask(
    val zoneId: Long,
    val snapshotRecord: StoredSnapshotRecord,
)

private const val DEFAULT_SEGMENTATION_WORKER_COUNT = 3
