package anton.axenov

import android.os.SystemClock
import android.view.Surface
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.math.Position
import io.github.sceneview.utils.worldToScreen
import java.util.IdentityHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import korlibs.math.geom.Vector3F as Vector3

/**
 * Coordinates end-to-end detection and placement for AR session updates.
 *
 * @param coroutineScope scope used to run asynchronous detection work.
 * @param reportStatus callback used to publish user-visible diagnostics.
 * @param segmentationServerClient client used to upload snapshots and fetch info from server.
 * @param zoneDetector detector used to extract interest zones from snapshots.
 * @param onTranslationInfoChanged callback invoked when translation info for overlay changes.
 * @param onMergeInfoChanged callback invoked when persistent merge diagnostics should be updated.
 * @param onUploadQueueInfoChanged callback invoked when upload queue diagnostics should be updated.
 * @param onZonesDetected callback invoked when zones are detected with optional frame projection context.
 * @param onZoneScreenLabelsChanged callback invoked when projected on-screen zone labels should be updated.
 */
class ArDetectionPipeline(
    private val coroutineScope: CoroutineScope,
    private val reportStatus: (message: String, force: Boolean) -> Unit,
    private val segmentationServerClient: SegmentationServerClient,
    private val zoneDetector: DetectInterestZones = DetectInterestZones(segmentationServerClient),
    private val onTranslationInfoChanged: (TranslationOverlayInfo) -> Unit = {},
    private val onMergeInfoChanged: (String) -> Unit = {},
    private val onUploadQueueInfoChanged: (String) -> Unit = {},
    private val onZonesDetected: (snapshot: DetectionFrameSnapshot, zones: List<DetectedInterestZone>) -> Unit = { _, _ -> },
    private val onZoneScreenLabelsChanged: (List<ZoneScreenLabelEntry>) -> Unit = {},
) {
    private val isSceneActive = AtomicBoolean(false)
    private var sceneView: ARSceneView? = null
    private val snapshotsManager = SnapshotsManager(
        onSnapshotStored = ::onZoneSnapshotStored,
        onSnapshotRemoved = ::onZoneSnapshotRemoved,
    )
    private val zonesManager = ZonesManager(
        onZoneAddition = ::onZoneAdded,
        onZoneDeletion = ::onZoneDeleted,
    )
    private val renderedNodesByZone = IdentityHashMap<Zone, MutableList<AnchorNode>>()
    private val renderedSnapshotNodesByZone = IdentityHashMap<Zone, IdentityHashMap<ZoneSnapshot, AnchorNode>>()
    private var detectionJob: Job? = null
    private var serverStatusJob: Job? = null
    private var lastDetectionAtMs = 0L
    private var lastMetricsLabelRefreshAtMs = 0L
    private var lastServerStatusRefreshAtMs = 0L
    private var lastCaptureFailureAtMs = 0L
    private var lastDepthUnavailableLogAtMs = 0L
    private var asyncPlacementNoticeShown = false
    private var lastTrackingState: TrackingState? = null
    private var lastStatusAtMs = 0L
    private var baselineLandscapeRotation: Int? = null
    private var lastTranslationVariant: CoordinateTranslationVariant? = null
    private val snapshotUploadQueue = SnapshotUploadQueue(
        coroutineScope = coroutineScope,
        segmentationServerClient = segmentationServerClient,
        onQueueInfoChanged = onUploadQueueInfoChanged,
        onUploadFailure = { zoneId, error ->
            reportStatus(
                "Server upload failed for zone $zoneId: ${error.message ?: error.javaClass.simpleName}",
                true,
            )
        },
    )

    /**
     * Registers active SceneView instance for subsequent session updates.
     *
     * @param sceneView created SceneView instance.
     */
    fun onSceneCreated(sceneView: ARSceneView) {
        this.sceneView = sceneView
        isSceneActive.set(true)
        snapshotUploadQueue.start()
    }

    /**
     * Cleans up detection jobs and placed anchors when scene is disposed.
     */
    fun onSceneDisposed() {
        isSceneActive.set(false)
        detectionJob?.cancel()
        serverStatusJob?.cancel()
        snapshotUploadQueue.stop()
        sceneView = null
        onZoneScreenLabelsChanged(emptyList())
        snapshotsManager.clear()
        zonesManager.clear()
        renderedNodesByZone.clear()
        renderedSnapshotNodesByZone.clear()
        baselineLandscapeRotation = null
    }

    /**
     * Handles one AR frame update and triggers snapshot/detect/place flow.
     *
     * @param frame current ARCore frame.
     */
    fun onSessionUpdated(frame: Frame) {
        if (!isSceneActive.get()) {
            return
        }
        val activeSceneView = sceneView ?: return

        if (!asyncPlacementNoticeShown) {
            asyncPlacementNoticeShown = true
            reportStatus(
                "Async placement uses captured frame depth snapshot; deferred ARCore hitTest on old frames is not reliable.",
                true,
            )
        }
        val trackingState = frame.camera.trackingState
        if (trackingState != lastTrackingState) {
            lastTrackingState = trackingState
            reportStatus("Camera tracking=${trackingState.name}", true)
        }

        val now = SystemClock.elapsedRealtime()
        if (now - lastStatusAtMs >= FRAME_STATUS_INTERVAL_MS) {
            lastStatusAtMs = now
            val sceneNodes = renderedNodesByZone.values.sumOf { nodes -> nodes.size }
            reportStatus(
                "Frame ts=${frame.timestamp}, " +
                        "tracking=${trackingState.name}, " +
                        "zonesPlaced=${renderedNodesByZone.size}, sceneNodes=${sceneNodes}, " +
                        "uploadQueue=${snapshotUploadQueue.queuedCount()}, " +
                        "uploadActive=${snapshotUploadQueue.activeCount()}",
                false,
            )
        }
        if (trackingState != TrackingState.TRACKING) {
            onZoneScreenLabelsChanged(emptyList())
            return
        }

        if (now - lastMetricsLabelRefreshAtMs >= METRICS_LABEL_REFRESH_INTERVAL_MS) {
            lastMetricsLabelRefreshAtMs = now
            val cameraPose = frame.camera.pose
            val cameraPosition = Vector3(
                x = cameraPose.tx(),
                y = cameraPose.ty(),
                z = cameraPose.tz(),
            )
            val worldPointProjector: (Vector3) -> ViewPoint? = { worldPoint ->
                val projectedPoint = activeSceneView.view.worldToScreen(
                    Position(worldPoint.x, worldPoint.y, worldPoint.z),
                )
                if (!projectedPoint.x.isFinite() || !projectedPoint.y.isFinite())
                    null
                else ViewPoint(
                    xPx = projectedPoint.x,
                    yPx = projectedPoint.y,
                )
            }
            zonesManager.refreshZoneMetricsLabels(
                cameraPosition = cameraPosition,
                screenWidth = activeSceneView.width,
                screenHeight = activeSceneView.height,
                worldPointProjector = worldPointProjector,
            )
            addZoneSnapshots(
                frame = frame,
                cameraPosition = cameraPosition,
                screenWidth = activeSceneView.width,
                screenHeight = activeSceneView.height,
                worldPointProjector = worldPointProjector,
            )
        }

        if (now - lastServerStatusRefreshAtMs >= SERVER_REFRESH_INTERVAL_MS && serverStatusJob?.isActive != true) {
            lastServerStatusRefreshAtMs = now
            requestServerUpdates()
        }

        publishZoneScreenLabels()

        if (now - lastDetectionAtMs >= DETECTION_INTERVAL_MS && detectionJob?.isActive != true) {
            val captureResult = captureDetectionFrameSnapshot(frame)
            val snapshot = captureResult.snapshot
            if (snapshot != null) {
                if (snapshot.depthSnapshot == null &&
                    now - lastDepthUnavailableLogAtMs >= CAPTURE_FAILURE_REPORT_INTERVAL_MS
                ) {
                    lastDepthUnavailableLogAtMs = now
                    reportStatus("Detection snapshot captured without depth: ${captureResult.details}", true)
                }
                lastDetectionAtMs = now
                detectionJob = coroutineScope.launch(Dispatchers.Default) {
                    try {
                        val translationVariant = when (val displayRotation = activeSceneView.display?.rotation) {
                            Surface.ROTATION_90,
                            Surface.ROTATION_270,
                                -> {
                                val baseline = baselineLandscapeRotation
                                if (baseline == null) {
                                    baselineLandscapeRotation = displayRotation
                                    CoordinateTranslationVariant.LANDSCAPE
                                } else if (displayRotation == baseline) {
                                    CoordinateTranslationVariant.LANDSCAPE
                                } else {
                                    CoordinateTranslationVariant.LANDSCAPE_REVERSED
                                }
                            }

                            else -> CoordinateTranslationVariant.PORTRAIT
                        }
                        if (translationVariant != lastTranslationVariant) {
                            lastTranslationVariant = translationVariant
                        }
                        onTranslationInfoChanged(
                            TranslationOverlayInfo(
                                translationVariant = translationVariant,
                                imageWidth = snapshot.imageWidth,
                                imageHeight = snapshot.imageHeight,
                                viewWidth = activeSceneView.width,
                                viewHeight = activeSceneView.height,
                            ),
                        )
                        val detectedZones = zoneDetector.detectZones(snapshot)
                        withContext(Dispatchers.Main.immediate) {
                            if (detectedZones.isEmpty()) {
                                reportStatus("No interest zones detected", false)
                                return@withContext
                            }
                            onZonesDetected(snapshot, detectedZones)
                            if (!isSceneActive.get() || sceneView !== activeSceneView) {
                                reportStatus(
                                    "Zone placement skipped: scene was recreated/disposed during async detection.",
                                    false,
                                )
                                return@withContext
                            }

                            detectedZones.forEachIndexed { index, detectedZone ->
                                if (!isSceneActive.get() || sceneView !== activeSceneView) {
                                    reportStatus(
                                        "Zone placement stopped: scene was recreated/disposed during async detection.",
                                        false,
                                    )
                                    return@forEachIndexed
                                }

                                val placementResult = runCatching {
                                    placeZoneInWorld(
                                        sceneView = activeSceneView,
                                        snapshot = snapshot,
                                        detectedZone = detectedZone,
                                        translationVariant = translationVariant,
                                    )
                                }.getOrElse { error ->
                                    ZonePlacementResult(
                                        zone = null,
                                        details =
                                            "Placement aborted: ${error.javaClass.simpleName}: " +
                                                    (error.message ?: "unknown error"),
                                    )
                                }
                                if (placementResult.zone != null) {
                                    zonesManager.addZones(listOf(placementResult.zone))
                                    val mergeInfo = zonesManager.consumeMergeDebugInfos()
                                    val mergeMessage = "Merge debug: $mergeInfo"
                                    onMergeInfoChanged(mergeMessage)
                                    reportStatus(
                                        mergeMessage,
                                        true,
                                    )
                                    removeQueuedZonesFromWorld()
                                    reportStatus(
                                        "Zone ${index + 1}/${detectedZones.size} placed using ${placementResult.details} " +
                                                "from frame ts=${snapshot.frameTimestamp}. ${placementResult.details}",
                                        true,
                                    )
                                } else {
                                    reportStatus(
                                        "Zone ${index + 1}/${detectedZones.size} placement failed. ${placementResult.details}",
                                        true,
                                    )
                                }
                            }
                            reportStatus(
                                "Processed ${detectedZones.size} detected zone(s) from frame ts=${snapshot.frameTimestamp}",
                                true,
                            )
                        }
                    } finally {
                        snapshot.screenshot.recycle()
                    }
                }
            } else if (now - lastCaptureFailureAtMs >= CAPTURE_FAILURE_REPORT_INTERVAL_MS) {
                lastCaptureFailureAtMs = now
                reportStatus("Detection snapshot skipped: ${captureResult.details}", false)
            }
        }
    }

    /**
     * Removes all queued zones from manager storage.
     */
    private fun removeQueuedZonesFromWorld() {
        val removedZones = zonesManager.consumeQueuedRemovedZones()
        if (removedZones.isEmpty()) {
            return
        }
        reportStatus("Removed ${removedZones.size} queued zone(s).", true)
    }

    /**
     * Draws all scene objects owned by one newly added zone.
     *
     * @param zone added zone that should be rendered.
     */
    private fun onZoneAdded(zone: Zone) {
        if (!isSceneActive.get()) {
            return
        }
        val activeSceneView = sceneView ?: return
        val nodes = drawZoneStaticNodes(
            sceneView = activeSceneView,
            zone = zone,
        )
        renderedNodesByZone[zone] = nodes.toMutableList()
    }

    /**
     * Destroys all rendered scene nodes owned by one removed zone.
     *
     * @param zone removed zone whose scene objects should be cleaned up.
     */
    private fun onZoneDeleted(zone: Zone) {
        snapshotsManager.removeZone(zone)
        renderedSnapshotNodesByZone.remove(zone)
        val zoneNodes = renderedNodesByZone.remove(zone) ?: return
        zoneNodes.forEach { node ->
            destroyAnchorNode(node)
        }
    }

    /**
     * Draws one direction marker for newly added zone snapshot.
     *
     * @param zone zone that owns the snapshot.
     * @param snapshot newly persisted snapshot.
     */
    private fun onZoneSnapshotStored(zone: Zone, snapshot: ZoneSnapshot) {
        if (!isSceneActive.get()) {
            return
        }
        val activeSceneView = sceneView ?: return
        drawSnapshotCameraDirectionNode(
            sceneView = activeSceneView,
            frameSnapshot = snapshot.frameSnapshot,
        )?.also { directionNode ->
            val zoneNodes = renderedSnapshotNodesByZone.getOrPut(zone) { IdentityHashMap() }
            zoneNodes[snapshot] = directionNode
        }
        if (!snapshotUploadQueue.enqueue(zone.id, snapshot.toPayload(zone))) {
            reportStatus("Server upload queue is unavailable for zone ${zone.id}", true)
        }
    }

    /**
     * Removes one rendered direction marker for removed zone snapshot.
     *
     * @param zone zone that owned removed snapshot.
     * @param snapshot removed persisted snapshot.
     */
    private fun onZoneSnapshotRemoved(zone: Zone, snapshot: ZoneSnapshot) {
        val zoneNodes = renderedSnapshotNodesByZone[zone] ?: return
        val node = zoneNodes.remove(snapshot) ?: return
        destroyAnchorNode(node)
        if (zoneNodes.isEmpty()) {
            renderedSnapshotNodesByZone.remove(zone)
        }
    }

    /**
     * Captures camera image for current frame and stores per-zone snapshots by angular uniqueness.
     *
     * @param frame current ARCore frame.
     * @param cameraPosition current camera world position.
     * @param screenWidth current view width.
     * @param screenHeight current view height.
     * @param worldPointProjector current world-to-screen projector.
     */
    private fun addZoneSnapshots(
        frame: Frame,
        cameraPosition: Vector3,
        screenWidth: Int,
        screenHeight: Int,
        worldPointProjector: (Vector3) -> ViewPoint?,
    ) {
        val zones = zonesManager.getZones()
        if (zones.isEmpty()) {
            return
        }
        val captureResult = captureDetectionFrameSnapshot(frame)
        val snapshot = captureResult.snapshot ?: return
        try {
            zones.forEach { zone ->
                val captureAngle = getZoneCaptureAngle(zone.planePose, cameraPosition)
                val screenCoverage = getZoneScreenCoverage(zone, screenWidth, screenHeight, worldPointProjector)
                snapshotsManager.addSnapshot(zone, snapshot, captureAngle, screenCoverage)
            }
        } finally {
            snapshot.screenshot.recycle()
        }
    }

    /**
     * Requests latest updates from server.
     */
    private fun requestServerUpdates() {
        serverStatusJob = coroutineScope.launch {
            val zoneTexts = runCatching {
                withContext(Dispatchers.IO) {
                    segmentationServerClient.fetchZoneTexts()
                }
            }.getOrElse { error ->
                reportStatus(
                    "Server status fetch failed: ${error.message ?: error.javaClass.simpleName}",
                    true,
                )
                return@launch
            }
            if (!isSceneActive.get()) {
                return@launch
            }
            val changedZones = zonesManager.applyServerTexts(zoneTexts)
            if (changedZones > 0) {
                publishZoneScreenLabels()
            }
        }
    }

    /**
     * Projects zone center points to current view and publishes visible label payloads for Compose overlay.
     *
     */
    private fun publishZoneScreenLabels() {
        val activeSceneView = sceneView ?: return
        if (activeSceneView.width <= 0 || activeSceneView.height <= 0) {
            onZoneScreenLabelsChanged(emptyList())
            return
        }
        val labels = zonesManager.getZones().mapNotNull { zone ->
            val projectedPoint = activeSceneView.view.worldToScreen(
                Position(zone.center.x, zone.center.y, zone.center.z),
            )
            if (!projectedPoint.x.isFinite() || !projectedPoint.y.isFinite()) {
                return@mapNotNull null
            }
            if (projectedPoint.x !in 0f..activeSceneView.width.toFloat() ||
                projectedPoint.y !in 0f..activeSceneView.height.toFloat()
            ) {
                return@mapNotNull null
            }
            ZoneScreenLabelEntry(
                text = zone.labelText,
                xPx = projectedPoint.x,
                yPx = projectedPoint.y,
            )
        }
        onZoneScreenLabelsChanged(labels)
    }

    /**
     * Safely removes one anchor node from scene graph and destroys AR resources.
     *
     * @param node anchor node to clean up.
     */
    private fun destroyAnchorNode(node: AnchorNode) {
        runCatching { node.parent?.removeChildNode(node) }
        runCatching { node.anchor.detach() }
        runCatching { node.destroy() }
    }
}

/**
 * Translation info displayed in debug overlay.
 *
 * @param translationVariant currently selected coordinate translation variant.
 * @param imageWidth captured camera image width in pixels.
 * @param imageHeight captured camera image height in pixels.
 * @param viewWidth AR view width in pixels.
 * @param viewHeight AR view height in pixels.
 */
data class TranslationOverlayInfo(
    val translationVariant: CoordinateTranslationVariant,
    val imageWidth: Int,
    val imageHeight: Int,
    val viewWidth: Int,
    val viewHeight: Int,
)

/**
 * One on-screen zone label tied to projected center of a 3D zone anchor point.
 *
 * @param text visible metrics text.
 * @param xPx horizontal screen coordinate in pixels.
 * @param yPx vertical screen coordinate in pixels.
 */
data class ZoneScreenLabelEntry(
    val text: String,
    val xPx: Float,
    val yPx: Float,
)

private const val DETECTION_INTERVAL_MS = 5000L
private const val CAPTURE_FAILURE_REPORT_INTERVAL_MS = 1500L
private const val FRAME_STATUS_INTERVAL_MS = 1000L
private const val METRICS_LABEL_REFRESH_INTERVAL_MS = 200L
private const val SERVER_REFRESH_INTERVAL_MS = 500L
