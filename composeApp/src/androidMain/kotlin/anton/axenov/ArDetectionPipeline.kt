package anton.axenov

import android.os.SystemClock
import android.view.Surface
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import java.util.Locale
import java.util.IdentityHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Coordinates end-to-end detection and placement for AR session updates.
 *
 * @param coroutineScope scope used to run asynchronous detection work.
 * @param reportStatus callback used to publish user-visible diagnostics.
 * @param zoneDetector detector used to extract interest zones from snapshots.
 * @param onTranslationInfoChanged callback invoked when translation info for overlay changes.
 * @param onMergeInfoChanged callback invoked when persistent merge diagnostics should be updated.
 * @param onZonesDetected callback invoked when zones are detected with optional frame projection context.
 */
class ArDetectionPipeline(
    private val coroutineScope: CoroutineScope,
    private val reportStatus: (message: String, force: Boolean) -> Unit,
    private val zoneDetector: DetectInterestZones = DetectInterestZones(),
    private val onTranslationInfoChanged: (TranslationOverlayInfo) -> Unit = {},
    private val onMergeInfoChanged: (String) -> Unit = {},
    private val onZonesDetected: (snapshot: DetectionFrameSnapshot, zones: List<DetectedInterestZone>) -> Unit = { _, _ -> },
) {
    private val isSceneActive = AtomicBoolean(false)
    private var sceneView: ARSceneView? = null
    private val zonesManager = ZonesManager()
    private val placedSceneNodes = mutableListOf<AnchorNode>()
    private val renderedNodesByZone = IdentityHashMap<Zone, List<AnchorNode>>()
    private var detectionJob: Job? = null
    private var lastDetectionAtMs = 0L
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
        placedSceneNodes.forEach { node ->
            runCatching { node.destroy() }
        }
        placedSceneNodes.clear()
        renderedNodesByZone.clear()
        baselineLandscapeRotation = null
        zonesManager.clear()
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
        if (
            trackingState == TrackingState.TRACKING &&
            now - lastDetectionAtMs >= DETECTION_INTERVAL_MS &&
            detectionJob?.isActive != true
        ) {
            val captureResult = captureDetectionFrameSnapshot(frame)
            val snapshot = captureResult.snapshot
            if (snapshot != null) {
                if (
                    snapshot.depthSnapshot == null &&
                    now - lastDepthUnavailableLogAtMs >= CAPTURE_FAILURE_REPORT_INTERVAL_MS
                ) {
                    lastDepthUnavailableLogAtMs = now
                    reportStatus("Detection snapshot captured without depth: ${captureResult.details}", true)
                }
                lastDetectionAtMs = now
                val activeSceneView = sceneView ?: return
                detectionJob = coroutineScope.launch(Dispatchers.Default) {
                    try {
                        val translationVariant = when (val displayRotation = activeSceneView.display?.rotation) {
                            Surface.ROTATION_90,
                            Surface.ROTATION_270,
                            -> {
                                // First observed landscape side is treated as baseline; opposite side means 180-degree rotation.
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

                            detectedZones.forEachIndexed { index, zone ->
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
                                        zone = zone,
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
                                    val undrawnZones = zonesManager.consumeUndrawnZones()
                                    val renderedNodes = undrawnZones.flatMap { drawnZone ->
                                        val drawnNodes = drawZone(
                                            sceneView = activeSceneView,
                                            zone = drawnZone,
                                        )
                                        renderedNodesByZone[drawnZone] = drawnNodes
                                        drawnNodes
                                    }
                                    placedSceneNodes += renderedNodes
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
            reportStatus(
                "Frame ts=${frame.timestamp}, tracking=${trackingState.name}, zonesPlaced=${renderedNodesByZone.size}, sceneNodes=${placedSceneNodes.size}",
                false,
            )
        }
    }

    /**
     * Removes all queued zones and destroys corresponding rendered nodes.
     */
    private fun removeQueuedZonesFromWorld() {
        val removedZones = zonesManager.consumeQueuedRemovedZones()
        if (removedZones.isEmpty()) {
            return
        }
        var removedNodesCount = 0
        var removedZonesWithoutSceneNodes = 0
        removedZones.forEach { zone ->
            val zoneNodes = renderedNodesByZone.remove(zone).orEmpty()
            if (zoneNodes.isEmpty()) {
                removedZonesWithoutSceneNodes++
            }
            zoneNodes.forEach { node ->
                runCatching { node.destroy() }
            }
            placedSceneNodes.removeAll(zoneNodes.toSet())
            removedNodesCount += zoneNodes.size
        }
        reportStatus(
            "Removed ${removedZones.size} zone(s), $removedNodesCount scene node(s), " +
                    "missingSceneMappings=$removedZonesWithoutSceneNodes.",
            true,
        )
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

private const val DETECTION_INTERVAL_MS = 10000L
private const val CAPTURE_FAILURE_REPORT_INTERVAL_MS = 1500L
private const val FRAME_STATUS_INTERVAL_MS = 1000L
