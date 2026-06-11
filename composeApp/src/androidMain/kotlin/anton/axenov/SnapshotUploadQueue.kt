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
    private val queuedUploads = mutableListOf<QueuedSnapshotUpload>()
    private var uploadSignals = Channel<Unit>(Channel.UNLIMITED)
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
                    uploadSignals.receiveCatching().getOrNull() ?: break
                    val task = synchronized(this@SnapshotUploadQueue) {
                        queuedUploads.removeFirstOrNullCompat()
                    } ?: run {
                        publishQueueInfo()
                        continue
                    }
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
        uploadSignals.close()
        uploadSignals = Channel(Channel.UNLIMITED)
        queuedUploadCount.set(0)
        activeUploadCount.set(0)
        synchronized(this) {
            queuedUploads.clear()
        }
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
        synchronized(this) {
            queuedUploads += QueuedSnapshotUpload(
                zoneId = zoneId,
                payload = payload,
            )
        }
        queuedUploadCount.incrementAndGet()
        val result = uploadSignals.trySend(Unit)
        if (result.isFailure) {
            synchronized(this) {
                queuedUploads.removeFirstMatching { queuedUpload ->
                    queuedUpload.payload.sessionId == payload.sessionId &&
                        queuedUpload.payload.requestId == payload.requestId
                }
            }
            queuedUploadCount.decrementAndGet()
            publishQueueInfo()
            return false
        }
        publishQueueInfo()
        return true
    }

    /**
     * Removes one upload that has not started yet.
     *
     * @param sessionId stable processing session identifier.
     * @param requestId logical request identifier.
     * @return true when the upload was still queued and is now cancelled.
     */
    fun cancel(
        sessionId: String,
        requestId: String,
    ): Boolean {
        val removed = synchronized(this) {
            queuedUploads.removeFirstMatching { queuedUpload ->
                queuedUpload.payload.sessionId == sessionId &&
                    queuedUpload.payload.requestId == requestId
            }
        }
        if (removed) {
            queuedUploadCount.decrementAndGet()
            publishQueueInfo()
        }
        return removed
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

/**
 * Removes the first element matching [predicate].
 *
 * @param predicate match condition.
 * @return true when one element was removed.
 */
private inline fun <T> MutableList<T>.removeFirstMatching(
    predicate: (T) -> Boolean,
): Boolean {
    val index = indexOfFirst(predicate)
    if (index < 0) {
        return false
    }
    removeAt(index)
    return true
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

private const val DEFAULT_UPLOAD_WORKER_COUNT = 3
