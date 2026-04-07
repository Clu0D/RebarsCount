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
 * @param zoneDetector detector used to extract interest zones from snapshots.
 * @param onTranslationInfoChanged callback invoked when translation info for overlay changes.
 * @param onMergeInfoChanged callback invoked when persistent merge diagnostics should be updated.
 * @param onZonesDetected callback invoked when zones are detected with optional frame projection context.
 * @param onZoneScreenLabelsChanged callback invoked when projected on-screen zone labels should be refreshed.
 */
class ArDetectionPipeline(
    private val coroutineScope: CoroutineScope,
    private val reportStatus: (message: String, force: Boolean) -> Unit,
    private val zoneDetector: DetectInterestZones = DetectInterestZones(),
    private val onTranslationInfoChanged: (TranslationOverlayInfo) -> Unit = {},
    private val onMergeInfoChanged: (String) -> Unit = {},
    private val onZonesDetected: (snapshot: DetectionFrameSnapshot, zones: List<DetectedInterestZone>) -> Unit = { _, _ -> },
    private val onZoneScreenLabelsChanged: (List<ZoneScreenLabelEntry>) -> Unit = {},
) {
    private val isSceneActive = AtomicBoolean(false)
    private var sceneView: ARSceneView? = null
    private val zonesManager = ZonesManager(
        onZoneAddition = ::onZoneAdded,
        onZoneDeletion = ::onZoneDeleted,
    )
    private val renderedNodesByZone = IdentityHashMap<Zone, MutableList<AnchorNode>>()
    private var detectionJob: Job? = null
    private var lastDetectionAtMs = 0L
    private var lastMetricsLabelRefreshAtMs = 0L
    private var lastCaptureFailureAtMs = 0L
    private var lastDepthUnavailableLogAtMs = 0L
    private var asyncPlacementNoticeShown = false
    private var lastTrackingState: TrackingState? = null
    private var lastStatusAtMs = 0L
    private var baselineLandscapeRotation: Int? = null
    private var lastTranslationVariant: CoordinateTranslationVariant? = null

    /**
     * Registers active SceneView instance for subsequent session updates.
     *
     * @param sceneView created SceneView instance.
     */
    fun onSceneCreated(sceneView: ARSceneView) {
        this.sceneView = sceneView
        isSceneActive.set(true)
    }

    /**
     * Cleans up detection jobs and placed anchors when scene is disposed.
     */
    fun onSceneDisposed() {
        isSceneActive.set(false)
        detectionJob?.cancel()
        sceneView = null
        onZoneScreenLabelsChanged(emptyList())
        zonesManager.clear()
        renderedNodesByZone.clear()
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
        if (trackingState == TrackingState.TRACKING &&
            now - lastMetricsLabelRefreshAtMs >= METRICS_LABEL_REFRESH_INTERVAL_MS
        ) {
            lastMetricsLabelRefreshAtMs = now
            val cameraPose = frame.camera.pose
            zonesManager.refreshZoneMetricsLabels(
                cameraPosition = Vector3(
                    x = cameraPose.tx(),
                    y = cameraPose.ty(),
                    z = cameraPose.tz(),
                ),
                screenWidth = activeSceneView.width,
                screenHeight = activeSceneView.height,
                worldPointProjector = { worldPoint ->
                    val projectedPoint = activeSceneView.view.worldToScreen(
                        Position(worldPoint.x, worldPoint.y, worldPoint.z),
                    )
                    if (!projectedPoint.x.isFinite() || !projectedPoint.y.isFinite())
                        null
                    else ViewPoint(
                        xPx = projectedPoint.x,
                        yPx = projectedPoint.y,
                    )
                },
            )
        }
        if (trackingState == TrackingState.TRACKING) {
            publishZoneScreenLabels()
        } else {
            onZoneScreenLabelsChanged(emptyList())
        }
        if (trackingState == TrackingState.TRACKING &&
            now - lastDetectionAtMs >= DETECTION_INTERVAL_MS &&
            detectionJob?.isActive != true
        ) {
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
                        val detectedZones = zoneDetector.detectZones(
                            screenshot = snapshot.screenshot
                        )
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

        if (now - lastStatusAtMs >= FRAME_STATUS_INTERVAL_MS) {
            lastStatusAtMs = now
            val sceneNodes = renderedNodesByZone.values.sumOf { nodes -> nodes.size }
            reportStatus(
                "Frame ts=${frame.timestamp}, tracking=${trackingState.name}, zonesPlaced=${renderedNodesByZone.size}, sceneNodes=${sceneNodes}",
                false,
            )
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
        val zoneNodes = renderedNodesByZone.remove(zone) ?: return
        zoneNodes.forEach { node ->
            destroyAnchorNode(node)
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
