package anton.axenov

import com.google.ar.core.Plane
import io.github.sceneview.ar.ARSceneView
import kotlin.random.Random
import korlibs.math.geom.Vector3F as Vector3

/**
 * Places a detected zone in world space using depth from the captured frame snapshot.
 *
 * @param sceneView active SceneView AR view.
 * @param snapshot immutable snapshot captured from a specific frame.
 * @param detectedZone that belongs to the same snapshot screenshot.
 * @param translationVariant detector-to-image translation variant used for sampled points.
 * @return zone placement result with diagnostics.
 */
fun placeZoneInWorld(
    sceneView: ARSceneView,
    snapshot: DetectionFrameSnapshot,
    detectedZone: DetectedInterestZone,
    translationVariant: CoordinateTranslationVariant,
): ZonePlacementResult {
    val imagePoints = sampleImagePointsInScreenBoundingBox(
        screenBoundingBox = detectedZone.screenBoundingBox,
        imageWidth = snapshot.imageWidth,
        imageHeight = snapshot.imageHeight,
        count = DEPTH_SAMPLE_POINT_COUNT,
        random = Random(snapshot.frameTimestamp),
    ).map { point ->
        translateCoordinates(
            x = point.x,
            y = point.y,
            width = snapshot.imageWidth,
            height = snapshot.imageHeight,
            translationVariant = translationVariant,
        )
    }

    val viewSamplePoints = sampleViewPointsInZone(
        snapshot = snapshot,
        imagePoints = imagePoints,
        sceneView = sceneView,
    )
    if (viewSamplePoints == null) {
        return failedPlacement(
            details = "Depth failed. Sample View points could not be found.",
        )
    }
    val pointPlacements = imagePoints.zip(viewSamplePoints).map { (imagePoint, viewPoint) ->
        placePointInWorld(
            sceneView = sceneView,
            snapshot = snapshot,
            imagePoint = imagePoint,
            viewPoint = viewPoint,
        )
    }
    val depthPlaced = pointPlacements.count { it?.method == PointPlacementMethod.DEPTH }
    val hitPlaced = pointPlacements.count { it?.method == PointPlacementMethod.HIT }
    val featurePlaced = pointPlacements.count { it?.method == PointPlacementMethod.FEATURE }
    val missed = pointPlacements.count { it == null }
    val placementSummary =
        "Points placement: depth - $depthPlaced, hit - $hitPlaced, feature - $featurePlaced, missed - $missed"

    val worldPoints = pointPlacements.mapNotNull { it?.worldPoint }
    if (worldPoints.size < PLANE_MIN_POINT_COUNT) {
        return failedPlacement(
            details =
                "$placementSummary. " +
                        "Need at least $PLANE_MIN_POINT_COUNT points, got ${worldPoints.size}/${pointPlacements.size}.",
        )
    }

    val cameraPosition = currentCameraPosition(sceneView) ?: snapshotCameraPosition(snapshot)
    val planeFit = fitPlanePoseFromPoints(
        worldPoints = worldPoints,
        cameraPosition = cameraPosition,
        minPointCount = PLANE_MIN_POINT_COUNT,
    )
    val fittedPlanePose = planeFit.pose
        ?: return failedPlacement("$placementSummary. Plane fit failed: ${planeFit.details}")

    val projectionInput = buildZoneProjectionInput(
        snapshot = snapshot,
        zone = detectedZone,
        translationVariant = translationVariant,
    )
    val zone = Zone(
        sampledPoints = worldPoints,
        planePose = fittedPlanePose,
        projectionInputs = listOf(projectionInput),
    )
    return ZonePlacementResult(
        zone = zone,
        details = "$placementSummary. " +
                "Estimated depth=${planeFit.depthMeters ?: -1f}m. " +
                "Plane fit: ${planeFit.details}. " +
                "Note: hit tests use current frame, not captured frame ${snapshot.frameTimestamp}." +
                "projectedCorners=${zone.polygonPoints.size}, " +
                "sampledPoints=${worldPoints.size}",
    )
}

/**
 * Resolves one sampled point into world space using three fallback methods in order.
 *
 * Order: depth projection from snapshot, plane hit test, feature-point hit test.
 *
 * @param sceneView active SceneView AR view.
 * @param snapshot immutable snapshot captured from a specific frame.
 * @param imagePoint sampled point in image coordinates.
 * @param viewPoint sampled point in SceneView coordinates.
 * @return resolved placement payload or null when all methods fail.
 */
private fun placePointInWorld(
    sceneView: ARSceneView,
    snapshot: DetectionFrameSnapshot,
    imagePoint: ImagePoint,
    viewPoint: ViewPoint,
): PointPlacement? {
    return placePointWithDepth(snapshot = snapshot, imagePoint = imagePoint)
        ?: placePointWithHit(sceneView = sceneView, viewPoint = viewPoint)
        ?: placePointWithFeature(sceneView = sceneView, viewPoint = viewPoint)
}

/**
 * Tries to place one sampled point using snapshot depth projection.
 *
 * @param snapshot immutable frame snapshot.
 * @param imagePoint sampled image-space point.
 * @return placement payload or null when depth projection fails.
 */
private fun placePointWithDepth(
    snapshot: DetectionFrameSnapshot,
    imagePoint: ImagePoint,
): PointPlacement? {
    val clampedX = imagePoint.x.coerceIn(0, snapshot.imageWidth - 1)
    val clampedY = imagePoint.y.coerceIn(0, snapshot.imageHeight - 1)
    val depthSample = sampleDepthMeters(snapshot, clampedX, clampedY)
    val depthMeters = depthSample.depthMeters
        ?: return null

    val xCamera = (clampedX - snapshot.principalPointX) / snapshot.focalLengthX * depthMeters
    val yCamera = -(clampedY - snapshot.principalPointY) / snapshot.focalLengthY * depthMeters
    val zCamera = -depthMeters
    val worldPoint = snapshot.cameraPose.transformPoint(floatArrayOf(xCamera, yCamera, zCamera))

    return PointPlacement(
        worldPoint = Vector3(worldPoint[0], worldPoint[1], worldPoint[2]),
        method = PointPlacementMethod.DEPTH,
    )
}

/**
 * Tries to place one sampled point using ARCore plane hit test.
 *
 * @param sceneView active SceneView AR view.
 * @param viewPoint sampled SceneView point.
 * @return placement payload or null when hit test fails.
 */
private fun placePointWithHit(
    sceneView: ARSceneView,
    viewPoint: ViewPoint,
): PointPlacement? {
    val hitPose = sceneView.hitTestAR(
        xPx = viewPoint.xPx,
        yPx = viewPoint.yPx,
        planeTypes = setOf(
            Plane.Type.HORIZONTAL_UPWARD_FACING,
            Plane.Type.HORIZONTAL_DOWNWARD_FACING,
            Plane.Type.VERTICAL,
        ),
    )?.hitPose ?: return null
    return PointPlacement(
        worldPoint = Vector3(
            x = hitPose.tx(),
            y = hitPose.ty(),
            z = hitPose.tz(),
        ),
        method = PointPlacementMethod.HIT,
    )
}

/**
 * Tries to place one sampled point using ARCore feature-point hit test.
 *
 * @param sceneView active SceneView AR view.
 * @param viewPoint sampled SceneView point.
 * @return placement payload or null when feature-point hit test fails.
 */
private fun placePointWithFeature(
    sceneView: ARSceneView,
    viewPoint: ViewPoint,
): PointPlacement? {
    val hitPose = sceneView.hitTestAR(
        xPx = viewPoint.xPx,
        yPx = viewPoint.yPx,
        point = true,
        depthPoint = false,
    )?.hitPose ?: return null
    return PointPlacement(
        worldPoint = Vector3(
            x = hitPose.tx(),
            y = hitPose.ty(),
            z = hitPose.tz(),
        ),
        method = PointPlacementMethod.FEATURE,
    )
}

/**
 * Builds a failed placement result with provided diagnostics.
 *
 * @param details detailed failure reason.
 * @return zone placement result with `FAILED` strategy.
 */
private fun failedPlacement(details: String): ZonePlacementResult {
    return ZonePlacementResult(
        zone = null,
        details = details,
    )
}


/**
 * Builds full zone projection input payload from captured snapshot and detected screen zone.
 *
 * @param snapshot captured frame snapshot.
 * @param zone detected zone in detector coordinates.
 * @param translationVariant detector-to-image translation variant.
 * @return projection payload that can be reused for zone reprojection.
 */
private fun buildZoneProjectionInput(
    snapshot: DetectionFrameSnapshot,
    zone: DetectedInterestZone,
    translationVariant: CoordinateTranslationVariant,
): ZoneProjectionInput {
    val cameraToWorldMatrix = FloatArray(CAMERA_MATRIX_SIZE).also { matrix ->
        snapshot.cameraPose.toMatrix(matrix, 0)
    }
    return ZoneProjectionInput(
        originalScreenPolygon = listOf(
            ImagePoint(zone.screenBoundingBox.left, zone.screenBoundingBox.top),
            ImagePoint(zone.screenBoundingBox.right, zone.screenBoundingBox.top),
            ImagePoint(zone.screenBoundingBox.right, zone.screenBoundingBox.bottom),
            ImagePoint(zone.screenBoundingBox.left, zone.screenBoundingBox.bottom),
        ),
        translationVariant = translationVariant,
        imageWidth = snapshot.imageWidth,
        imageHeight = snapshot.imageHeight,
        focalLengthX = snapshot.focalLengthX,
        focalLengthY = snapshot.focalLengthY,
        principalPointX = snapshot.principalPointX,
        principalPointY = snapshot.principalPointY,
        cameraToWorldMatrix = cameraToWorldMatrix,
    )
}


/**
 * Samples depth in meters from the snapshot depth map at image-space coordinate.
 *
 * @param snapshot frame snapshot with optional depth map.
 * @param imageX X coordinate in screenshot/image pixels.
 * @param imageY Y coordinate in screenshot/image pixels.
 * @return depth sampling attempt with optional depth and diagnostics.
 */
private fun sampleDepthMeters(
    snapshot: DetectionFrameSnapshot,
    imageX: Int,
    imageY: Int,
): DepthSampleAttempt {
    val depthSnapshot = snapshot.depthSnapshot
        ?: return DepthSampleAttempt(
            depthMeters = null,
            details = "Depth snapshot is unavailable for frame ts=${snapshot.frameTimestamp}",
        )
    if (depthSnapshot.width <= 0 || depthSnapshot.height <= 0) {
        return DepthSampleAttempt(
            depthMeters = null,
            details = "Depth map has invalid size ${depthSnapshot.width}x${depthSnapshot.height}",
        )
    }

    val centerDepthX = ((imageX.toFloat() / snapshot.imageWidth) * depthSnapshot.width)
        .toInt()
        .coerceIn(0, depthSnapshot.width - 1)
    val centerDepthY = ((imageY.toFloat() / snapshot.imageHeight) * depthSnapshot.height)
        .toInt()
        .coerceIn(0, depthSnapshot.height - 1)

    var bestConfidence = -1
    val bestConfidenceCandidates = mutableListOf<DepthCandidate>()
    var validDepthCount = 0
    var positiveConfidenceCount = 0

    for (dy in -DEPTH_SAMPLE_RADIUS_PX..DEPTH_SAMPLE_RADIUS_PX) {
        for (dx in -DEPTH_SAMPLE_RADIUS_PX..DEPTH_SAMPLE_RADIUS_PX) {
            val sampleX = (centerDepthX + dx).coerceIn(0, depthSnapshot.width - 1)
            val sampleY = (centerDepthY + dy).coerceIn(0, depthSnapshot.height - 1)
            val rawDepth = depthSnapshot.values[sampleY * depthSnapshot.width + sampleX].toInt() and 0xFFFF
            val depthMillimeters = rawDepth and DEPTH16_DEPTH_MASK
            val confidence = (rawDepth and DEPTH16_CONFIDENCE_MASK) ushr DEPTH16_CONFIDENCE_SHIFT
            if (depthMillimeters > 0) {
                validDepthCount++
                if (confidence > 0) {
                    positiveConfidenceCount++
                }
                if (confidence > bestConfidence) {
                    bestConfidence = confidence
                    bestConfidenceCandidates.clear()
                    bestConfidenceCandidates += DepthCandidate(
                        x = sampleX,
                        y = sampleY,
                        depthMillimeters = depthMillimeters,
                    )
                } else if (confidence == bestConfidence) {
                    bestConfidenceCandidates += DepthCandidate(
                        x = sampleX,
                        y = sampleY,
                        depthMillimeters = depthMillimeters,
                    )
                }
            }
        }
    }

    val medianCandidate = selectMedianPointCandidate(bestConfidenceCandidates)
        ?: return DepthSampleAttempt(
            depthMeters = null,
            details =
                "No valid depth in neighborhood. " +
                        "image($imageX,$imageY) -> centerDepth($centerDepthX,$centerDepthY), " +
                        "windowRadius=$DEPTH_SAMPLE_RADIUS_PX, validDepthCount=$validDepthCount",
        )

    return DepthSampleAttempt(
        depthMeters = medianCandidate.depthMillimeters / MILLIMETERS_IN_METER,
        details =
            "image($imageX,$imageY) -> bestDepth(${medianCandidate.x},${medianCandidate.y}), " +
                    "windowRadius=$DEPTH_SAMPLE_RADIUS_PX, depthMm=${medianCandidate.depthMillimeters}, " +
                    "confidence=$bestConfidence, validDepthCount=$validDepthCount, " +
                    "positiveConfidenceCount=$positiveConfidenceCount, tieCandidates=${bestConfidenceCandidates.size}, " +
                    "tieBreak=medianPoint",
    )
}

/**
 * Samples random SceneView points inside zone and always includes center.
 *
 * @param snapshot frame snapshot.
 * @param sceneView scene view dimensions.
 * @param count number of points to sample.
 * @return list of SceneView coordinates or null if SceneView dimensions are invalid.
 */
private fun sampleViewPointsInZone(
    snapshot: DetectionFrameSnapshot,
    imagePoints: List<ImagePoint>,
    sceneView: ARSceneView,
): List<ViewPoint>? {
    if (sceneView.width <= 0 || sceneView.height <= 0) {
        return null
    }
    return mapImagePointsToViewPoints(
        imagePoints = imagePoints,
        imageWidth = snapshot.imageWidth,
        imageHeight = snapshot.imageHeight,
        viewWidth = sceneView.width,
        viewHeight = sceneView.height,
    )
}

/**
 * Returns current camera position in world coordinates.
 *
 * @param sceneView active SceneView.
 * @return camera position or null when frame is unavailable.
 */
private fun currentCameraPosition(sceneView: ARSceneView): Vector3? {
    val pose = sceneView.frame?.camera?.pose ?: return null
    return Vector3(pose.tx(), pose.ty(), pose.tz())
}

/**
 * Returns snapshot camera position in world coordinates.
 *
 * @param snapshot captured frame snapshot.
 * @return camera position vector.
 */
private fun snapshotCameraPosition(snapshot: DetectionFrameSnapshot): Vector3 {
    return Vector3(
        snapshot.cameraPose.tx(),
        snapshot.cameraPose.ty(),
        snapshot.cameraPose.tz(),
    )
}

/**
 * Point placement method used in mixed fallback pipeline.
 */
private enum class PointPlacementMethod {
    DEPTH,
    HIT,
    FEATURE,
}

/**
 * Point placement payload returned by one-point fallback methods.
 *
 * @param worldPoint resolved point in world coordinates.
 * @param method method used to resolve this point.
 */
private data class PointPlacement(
    val worldPoint: Vector3,
    val method: PointPlacementMethod,
)

/**
 * Result of one depth sampling attempt.
 *
 * @param depthMeters sampled depth in meters or null when unavailable.
 * @param details detailed depth sampling diagnostics.
 */
private data class DepthSampleAttempt(
    val depthMeters: Float?,
    val details: String,
)

private const val DEPTH16_DEPTH_MASK = 0x1FFF
private const val DEPTH16_CONFIDENCE_MASK = 0xE000
private const val DEPTH16_CONFIDENCE_SHIFT = 13
private const val MILLIMETERS_IN_METER = 1000f
private const val DEPTH_SAMPLE_RADIUS_PX = 4
private const val DEPTH_SAMPLE_POINT_COUNT = 20
private const val PLANE_MIN_POINT_COUNT = 6
private const val CAMERA_MATRIX_SIZE = 16
