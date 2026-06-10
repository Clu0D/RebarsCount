package anton.axenov

import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

/**
 * Uploads snapshot payloads in a background queue.
 *
 * @param coroutineScope scope used to host worker coroutines and UI callbacks.
 * @param segmentationServerClient client used to upload queued payloads.
 * @param workerCount number of parallel upload workers.
 * @param onQueueInfoChanged callback invoked when queue diagnostics change.
 * @param onUploadFailure callback invoked when upload fails.
 */
class SnapshotUploadQueue(
    private val coroutineScope: CoroutineScope,
    private val segmentationServerClient: SegmentationClient,
    private val workerCount: Int = DEFAULT_UPLOAD_WORKER_COUNT,
    private val onQueueInfoChanged: (String) -> Unit = {},
    private val onUploadFailure: (zoneId: Long, error: Throwable) -> Unit = { _, _ -> },
) {
    private var uploadQueue = Channel<QueuedSnapshotUpload>(Channel.UNLIMITED)
    private val uploadWorkerJobs = mutableListOf<Job>()
    private val queuedUploadCount = AtomicInteger(0)
    fun queuedCount(): Int = queuedUploadCount.get()
    private val activeUploadCount = AtomicInteger(0)
    fun activeCount(): Int = activeUploadCount.get()

    /**
     * Starts background upload workers.
     */
    fun start() {
        if (uploadWorkerJobs.isNotEmpty()) {
            return
        }
        repeat(workerCount) {
            uploadWorkerJobs += coroutineScope.launch(Dispatchers.IO) {
                while (true) {
                    val task = uploadQueue.receiveCatching().getOrNull() ?: break
                    queuedUploadCount.decrementAndGet()
                    activeUploadCount.incrementAndGet()
                    publishQueueInfo()
                    try {
                        segmentationServerClient.predictPoints(task.payload)
                    } catch (error: Exception) {
                        onUploadFailure(task.zoneId, error)
                    } finally {
                        activeUploadCount.decrementAndGet()
                        publishQueueInfo()
                    }
                }
            }
        }
        publishQueueInfo()
    }

    /**
     * Stops all workers and clears queued state.
     */
    fun stop() {
        uploadWorkerJobs.forEach { workerJob ->
            workerJob.cancel()
        }
        uploadWorkerJobs.clear()
        uploadQueue.close()
        uploadQueue = Channel(Channel.UNLIMITED)
        queuedUploadCount.set(0)
        activeUploadCount.set(0)
        publishQueueInfo()
    }

    /**
     * Queues one upload.
     *
     * @param zoneId identifier used in diagnostics.
     * @param payload snapshot payload to upload.
     * @return true when payload was queued.
     */
    fun enqueue(
        zoneId: Long,
        payload: ZoneSnapshotUploadDto,
    ): Boolean {
        queuedUploadCount.incrementAndGet()
        val result = uploadQueue.trySend(
            QueuedSnapshotUpload(
                zoneId = zoneId,
                payload = payload,
            ),
        )
        if (result.isFailure) {
            queuedUploadCount.decrementAndGet()
            publishQueueInfo()
            return false
        }
        publishQueueInfo()
        return true
    }

    /**
     * Pushes current queue diagnostics to UI callback.
     */
    private fun publishQueueInfo() {
        val infoText = "Upload queue: queued=${queuedUploadCount.get()}, active=${activeUploadCount.get()}"
        coroutineScope.launch(Dispatchers.Main.immediate) {
            onQueueInfoChanged(infoText)
        }
    }
}

private data class QueuedSnapshotUpload(
    val zoneId: Long,
    val payload: ZoneSnapshotUploadDto,
)

private const val DEFAULT_UPLOAD_WORKER_COUNT = 3
