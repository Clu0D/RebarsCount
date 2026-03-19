package anton.axenov

import android.os.SystemClock
import android.view.Surface
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
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
 * @param onTranslationVariantChanged callback invoked when translation variant changes.
 * @param onZonesDetected callback invoked when zones are detected with optional frame projection context.
 */
class ArDetectionPipeline(
    private val coroutineScope: CoroutineScope,
    private val reportStatus: (message: String, force: Boolean) -> Unit,
    private val zoneDetector: DetectInterestZones = DetectInterestZones(),
    private val onTranslationVariantChanged: (CoordinateTranslationVariant) -> Unit = {},
    private val onZonesDetected: (snapshot: DetectionFrameSnapshot, zones: List<DetectedInterestZone>) -> Unit = { _, _ -> },
) {
    private val isSceneActive = AtomicBoolean(false)
    private var sceneView: ARSceneView? = null
    private val placedSceneNodes = mutableListOf<AnchorNode>()
    private var placedZoneCount = 0
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
        placedZoneCount = 0
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
                            onTranslationVariantChanged(translationVariant)
                        }
                        val detectedZones = zoneDetector.detectZones(
                            screenshot = snapshot.screenshot,
                            translationVariant = translationVariant,
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
                                    placeBoundingBoxInWorld(
                                        sceneView = activeSceneView,
                                        snapshot = snapshot,
                                        zone = zone,
                                    )
                                }.getOrElse { error ->
                                    BoundingBoxPlacementResult(
                                        anchorNode = null,
                                        pointNodes = emptyList(),
                                        strategy = PlacementStrategy.FAILED,
                                        details =
                                            "Placement aborted: ${error.javaClass.simpleName}: " +
                                                (error.message ?: "unknown error"),
                                    )
                                }
                                if (placementResult.placedNodes.isNotEmpty()) {
                                    placedSceneNodes += placementResult.placedNodes
                                    if (placementResult.anchorNode != null) {
                                        placedZoneCount++
                                    }
                                    reportStatus(
                                        "Zone ${index + 1}/${detectedZones.size} placed using ${placementResult.strategy.name} " +
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
                "Frame ts=${frame.timestamp}, tracking=${trackingState.name}, zonesPlaced=$placedZoneCount, sceneNodes=${placedSceneNodes.size}",
                false,
            )
        }
    }
}

private const val DETECTION_INTERVAL_MS = 10000L
private const val CAPTURE_FAILURE_REPORT_INTERVAL_MS = 1500L
private const val FRAME_STATUS_INTERVAL_MS = 1000L
