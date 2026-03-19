package anton.axenov

import com.google.ar.core.Plane
import com.google.ar.core.Pose
import io.github.sceneview.ar.ARSceneView
import kotlin.random.Random
import korlibs.math.geom.Vector3F as Vector3


/**
 * Placement strategy used to turn a 2D bounding box into a world anchor.
 */
enum class PlacementStrategy {
    DEPTH_SNAPSHOT,
    PLANE_HIT,
    FEATURE_POINT_HIT,
    FAILED,
}

/**
 * Places a detected bounding box in world space using depth from the captured frame snapshot.
 *
 * @param sceneView active SceneView AR view.
 * @param snapshot immutable snapshot captured from a specific frame.
 * @param zone detected zone that belongs to the same snapshot screenshot.
 * @return placement result with strategy and diagnostics.
 */
fun placeBoundingBoxInWorld(
    sceneView: ARSceneView,
    snapshot: DetectionFrameSnapshot,
    zone: DetectedInterestZone,
): BoundingBoxPlacementResult {
    val depthProjectionAttempt = projectBoundingBoxCenterToWorld(snapshot, zone.boundingBox)
    if (depthProjectionAttempt.worldPose != null) {
        return placeFromPose(
            sceneView = sceneView,
            snapshot = snapshot,
            zone = zone,
            pose = depthProjectionAttempt.worldPose,
            worldPoints = depthProjectionAttempt.worldPoints,
            depthMeters = depthProjectionAttempt.depthMeters,
            strategy = PlacementStrategy.DEPTH_SNAPSHOT,
            sessionNullDetails = "Depth projection succeeded, but AR session is null",
            successDetailsPrefix = "Depth snapshot placement successful. ${depthProjectionAttempt.details}.",
        )
    }

    val viewSamplePoints = sampleViewPointsInBoundingBox(
        snapshot = snapshot,
        boundingBox = zone.boundingBox,
        sceneView = sceneView,
        count = HIT_TEST_SAMPLE_POINT_COUNT,
    )
    if (viewSamplePoints == null) {
        return failedPlacement(
            details = "Depth failed. ${depthProjectionAttempt.details}. Current scene size is not ready for fallback hit test.",
        )
    }

    val cameraPosition = currentCameraPosition(sceneView) ?: snapshotCameraPosition(snapshot)
    val planeHitPoints = collectHitPoints(viewSamplePoints) { xPx, yPx ->
        sceneView.hitTestAR(
            xPx = xPx,
            yPx = yPx,
            planeTypes = setOf(
                Plane.Type.HORIZONTAL_UPWARD_FACING,
                Plane.Type.HORIZONTAL_DOWNWARD_FACING,
                Plane.Type.VERTICAL,
            ),
        )?.hitPose
    }
    val planePlacementResult = placeFromFallbackHitPoints(
        sceneView = sceneView,
        snapshot = snapshot,
        zone = zone,
        hitPoints = planeHitPoints,
        cameraPosition = cameraPosition,
        strategy = PlacementStrategy.PLANE_HIT,
        fitFailureDetailsPrefix =
            "Depth failed. ${depthProjectionAttempt.details}. " +
                    "Plane hit points found=${planeHitPoints.size}, but plane fit failed",
        sessionNullDetails = "Depth failed and plane fit succeeded, but AR session is null",
        successDetailsPrefix =
            "Depth failed. ${depthProjectionAttempt.details}. " +
                    "Fallback plane hit succeeded with ${planeHitPoints.size}/${viewSamplePoints.size} points.",
    )
    if (planePlacementResult != null) {
        return planePlacementResult
    }

    val featureHitPoints = collectHitPoints(viewSamplePoints) { xPx, yPx ->
        sceneView.hitTestAR(
            xPx = xPx,
            yPx = yPx,
            point = true,
            depthPoint = false,
        )?.hitPose
    }
    val featurePlacementResult = placeFromFallbackHitPoints(
        sceneView = sceneView,
        snapshot = snapshot,
        zone = zone,
        hitPoints = featureHitPoints,
        cameraPosition = cameraPosition,
        strategy = PlacementStrategy.FEATURE_POINT_HIT,
        fitFailureDetailsPrefix =
            "Depth and plane fallback failed. " +
                    "Feature/depth-point hits found=${featureHitPoints.size}, but plane fit failed",
        sessionNullDetails = "Depth/feature fit succeeded, but AR session is null",
        successDetailsPrefix =
            "Depth and plane fallback failed. " +
                    "Feature/depth-point fallback succeeded with ${featureHitPoints.size}/${viewSamplePoints.size} points.",
    )
    if (featurePlacementResult != null) {
        return featurePlacementResult
    }

    return failedPlacement(
        details =
            "Depth failed. ${depthProjectionAttempt.details}. " +
                    "Plane hits=${planeHitPoints.size}/${viewSamplePoints.size}, " +
                    "feature/depth-point hits=${featureHitPoints.size}/${viewSamplePoints.size}.",
    )
}

/**
 * Builds a failed placement result with provided diagnostics.
 *
 * @param details detailed failure reason.
 * @return placement result with `FAILED` strategy.
 */
private fun failedPlacement(details: String): BoundingBoxPlacementResult {
    return BoundingBoxPlacementResult(
        anchorNode = null,
        pointNodes = emptyList(),
        strategy = PlacementStrategy.FAILED,
        details = details,
    )
}

/**
 * Collects world-space hit points for sampled view coordinates.
 *
 * @param viewSamplePoints sampled SceneView points.
 * @param hitPoseProvider callback that resolves a hit pose for one sampled point.
 * @return list of world points from successful hits.
 */
private fun collectHitPoints(
    viewSamplePoints: List<ViewPoint>,
    hitPoseProvider: (xPx: Float, yPx: Float) -> Pose?,
): List<Vector3> {
    val hitPoints = mutableListOf<Vector3>()
    viewSamplePoints.forEach { point ->
        val hitPose = hitPoseProvider(point.xPx, point.yPx)
        if (hitPose != null) {
            hitPoints += Vector3(
                x = hitPose.tx(),
                y = hitPose.ty(),
                z = hitPose.tz(),
            )
        }
    }
    return hitPoints
}

/**
 * Tries to fit a plane from fallback hit points and place a rectangle marker on it.
 *
 * @param sceneView active SceneView.
 * @param snapshot frame snapshot used for diagnostics and size estimation.
 * @param zone detected zone being placed.
 * @param hitPoints fallback world points gathered from hit tests.
 * @param cameraPosition camera position used to orient fitted plane.
 * @param strategy placement strategy to report on success.
 * @param fitFailureDetailsPrefix message prefix for plane-fit failure.
 * @param sessionNullDetails message returned when AR session is unavailable.
 * @param successDetailsPrefix message prefix for successful placement.
 * @return placement result, or null when there are not enough hit points for this fallback.
 */
private fun placeFromFallbackHitPoints(
    sceneView: ARSceneView,
    snapshot: DetectionFrameSnapshot,
    zone: DetectedInterestZone,
    hitPoints: List<Vector3>,
    cameraPosition: Vector3,
    strategy: PlacementStrategy,
    fitFailureDetailsPrefix: String,
    sessionNullDetails: String,
    successDetailsPrefix: String,
): BoundingBoxPlacementResult? {
    if (hitPoints.size < PLANE_MIN_POINT_COUNT) {
        return null
    }
    val planePoseAttempt = fitPlanePoseFromPoints(
        worldPoints = hitPoints,
        cameraPosition = cameraPosition,
        minPointCount = PLANE_MIN_POINT_COUNT,
    )
    val pose = planePoseAttempt.pose?.toArPose()
        ?: return failedPlacement("$fitFailureDetailsPrefix: ${planePoseAttempt.details}")

    return placeFromPose(
        sceneView = sceneView,
        snapshot = snapshot,
        zone = zone,
        pose = pose,
        worldPoints = hitPoints,
        depthMeters = planePoseAttempt.depthMeters,
        strategy = strategy,
        sessionNullDetails = sessionNullDetails,
        successDetailsPrefix =
            "$successDetailsPrefix " +
                    "Estimated depth=${planePoseAttempt.depthMeters ?: -1f}m. " +
                    "Plane fit: ${planePoseAttempt.details}. " +
                    "Note: fallback hit uses current frame, not captured frame ${snapshot.frameTimestamp}.",
    )
}

/**
 * Places a rectangle marker by creating an anchor from a world pose.
 *
 * @param sceneView active SceneView.
 * @param snapshot frame snapshot with camera intrinsics.
 * @param zone detected zone used for rectangle sizing.
 * @param pose world pose to anchor the rectangle to.
 * @param depthMeters estimated depth used for physical size estimation.
 * @param strategy strategy to store in placement result.
 * @param sessionNullDetails details message when AR session is unavailable.
 * @param successDetailsPrefix details prefix when placement succeeds.
 * @return placement result.
 */
private fun placeFromPose(
    sceneView: ARSceneView,
    snapshot: DetectionFrameSnapshot,
    zone: DetectedInterestZone,
    pose: Pose,
    worldPoints: List<Vector3>,
    depthMeters: Float?,
    strategy: PlacementStrategy,
    sessionNullDetails: String,
    successDetailsPrefix: String,
): BoundingBoxPlacementResult {
    val session = sceneView.session ?: return failedPlacement(sessionNullDetails)
    val anchor = session.createAnchor(pose)
    val rectangleSize = computeRectanglePhysicalSize(
        boundingBox = zone.boundingBox,
        depthMeters = depthMeters,
        focalLengthX = snapshot.focalLengthX,
        focalLengthY = snapshot.focalLengthY,
        minRectangleSizeMeters = MIN_RECTANGLE_SIZE_METERS,
        maxRectangleSizeMeters = MAX_RECTANGLE_SIZE_METERS,
        minDepthMeters = MIN_RECTANGLE_DEPTH_METERS,
        maxDepthMeters = MAX_RECTANGLE_DEPTH_METERS,
        defaultDepthMeters = DEFAULT_FALLBACK_DEPTH_METERS,
    )
    val anchorNode = createRectangleMarkerAnchorNode(
        sceneView = sceneView,
        anchor = anchor,
        rectangleWidthMeters = rectangleSize.first,
        rectangleHeightMeters = rectangleSize.second,
    )
    val pointNodes = createWorldPointMarkerNodes(
        sceneView = sceneView,
        worldPoints = worldPoints,
    )
    return BoundingBoxPlacementResult(
        anchorNode = anchorNode,
        pointNodes = pointNodes,
        strategy = strategy,
        details =
            "$successDetailsPrefix " +
                    "Rectangle size(m)=(${rectangleSize.first}, ${rectangleSize.second}), " +
                    "renderedPoints=${pointNodes.size}",
    )
}

/**
 * Projects a bounding-box center from 2D screenshot space to world coordinates.
 *
 * @param snapshot frame snapshot that owns screenshot and depth data.
 * @param boundingBox bounding box in snapshot screenshot pixel coordinates.
 * @return projection attempt result with optional world point and diagnostics.
 */
private fun projectBoundingBoxCenterToWorld(
    snapshot: DetectionFrameSnapshot,
    boundingBox: BoundingBox,
): DepthProjectionAttempt {
    val imageSamples = sampleImagePointsInBoundingBox(
        boundingBox = boundingBox,
        imageWidth = snapshot.imageWidth,
        imageHeight = snapshot.imageHeight,
        count = DEPTH_SAMPLE_POINT_COUNT,
        random = Random(snapshot.frameTimestamp),
    )
    val worldPoints = mutableListOf<Vector3>()
    val depthValues = mutableListOf<Float>()
    var sampleFailures = 0
    var lastFailureDetails = "none"

    imageSamples.forEach { sample ->
        val projection = projectImagePointToWorld(snapshot, sample.x, sample.y)
        if (projection.worldPoint != null && projection.depthMeters != null) {
            worldPoints += projection.worldPoint
            depthValues += projection.depthMeters
        } else {
            sampleFailures++
            lastFailureDetails = projection.details
        }
    }

    if (worldPoints.size < PLANE_MIN_POINT_COUNT) {
        return DepthProjectionAttempt(
            worldPose = null,
            depthMeters = null,
            worldPoints = worldPoints,
            details =
                "Depth points are insufficient: ${worldPoints.size}/${imageSamples.size}. " +
                        "Last failure: $lastFailureDetails",
        )
    }

    val planeFit = fitPlanePoseFromPoints(
        worldPoints = worldPoints,
        cameraPosition = snapshotCameraPosition(snapshot),
        minPointCount = PLANE_MIN_POINT_COUNT,
    )
    val planePose = planeFit.pose
        ?: return DepthProjectionAttempt(
            worldPose = null,
            depthMeters = null,
            worldPoints = worldPoints,
            details =
                "Depth points=${worldPoints.size}/${imageSamples.size}, " +
                        "plane fit failed: ${planeFit.details}",
        )

    val averageDepth = depthValues.average().toFloat()
    return DepthProjectionAttempt(
        worldPose = planePose.toArPose(),
        depthMeters = averageDepth,
        worldPoints = worldPoints,
        details =
            "Depth points=${worldPoints.size}/${imageSamples.size}, failures=$sampleFailures, " +
                    "avgDepth=$averageDepth. Plane fit: ${planeFit.details}",
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
 * Projects one image-space point to world coordinates using snapshot depth and intrinsics.
 *
 * @param snapshot captured frame snapshot.
 * @param imageX image-space X coordinate.
 * @param imageY image-space Y coordinate.
 * @return projection attempt result.
 */
private fun projectImagePointToWorld(
    snapshot: DetectionFrameSnapshot,
    imageX: Int,
    imageY: Int,
): ImagePointProjectionAttempt {
    val clampedX = imageX.coerceIn(0, snapshot.imageWidth - 1)
    val clampedY = imageY.coerceIn(0, snapshot.imageHeight - 1)
    val depthSample = sampleDepthMeters(snapshot, clampedX, clampedY)
    val depthMeters = depthSample.depthMeters
        ?: return ImagePointProjectionAttempt(
            worldPoint = null,
            depthMeters = null,
            details = "Depth sample failed at image($clampedX,$clampedY): ${depthSample.details}",
        )

    val xCamera = (clampedX - snapshot.principalPointX) / snapshot.focalLengthX * depthMeters
    val yCamera = -(clampedY - snapshot.principalPointY) / snapshot.focalLengthY * depthMeters
    val zCamera = -depthMeters
    val worldPoint = snapshot.cameraPose.transformPoint(floatArrayOf(xCamera, yCamera, zCamera))
    return ImagePointProjectionAttempt(
        worldPoint = Vector3(worldPoint[0], worldPoint[1], worldPoint[2]),
        depthMeters = depthMeters,
        details = "Point image($clampedX,$clampedY) -> camera($xCamera,$yCamera,$zCamera)",
    )
}

/**
 * Samples random SceneView points inside projected bounding box and always includes center.
 *
 * @param snapshot frame snapshot.
 * @param boundingBox bounding box in image coordinates.
 * @param sceneView scene view dimensions.
 * @param count number of points to sample.
 * @return list of SceneView coordinates or null if SceneView dimensions are invalid.
 */
private fun sampleViewPointsInBoundingBox(
    snapshot: DetectionFrameSnapshot,
    boundingBox: BoundingBox,
    sceneView: ARSceneView,
    count: Int,
): List<ViewPoint>? {
    if (sceneView.width <= 0 || sceneView.height <= 0) {
        return null
    }
    return mapImagePointsToViewPoints(
        imagePoints = sampleImagePointsInBoundingBox(
            boundingBox = boundingBox,
            imageWidth = snapshot.imageWidth,
            imageHeight = snapshot.imageHeight,
            count = count,
            random = Random(snapshot.frameTimestamp),
        ),
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
 * Converts shared pose representation to ARCore pose.
 *
 * @return ARCore pose.
 */
private fun PlanePoseData.toArPose(): Pose {
    return Pose(
        floatArrayOf(center.x, center.y, center.z),
        floatArrayOf(rotation.x, rotation.y, rotation.z, rotation.w),
    )
}

/**
 * Result of 2D-to-3D depth projection for a bounding box.
 *
 * @param worldPose projected world pose or null when projection failed.
 * @param depthMeters projected depth in meters.
 * @param details detailed projection diagnostics.
 */
private data class DepthProjectionAttempt(
    val worldPose: Pose?,
    val depthMeters: Float?,
    val worldPoints: List<Vector3>,
    val details: String,
)

/**
 * Result of image-point projection attempt.
 *
 * @param worldPoint projected world point or null when projection failed.
 * @param depthMeters sampled depth in meters.
 * @param details projection diagnostics.
 */
private data class ImagePointProjectionAttempt(
    val worldPoint: Vector3?,
    val depthMeters: Float?,
    val details: String,
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
private const val HIT_TEST_SAMPLE_POINT_COUNT = 20
private const val PLANE_MIN_POINT_COUNT = 6
private const val MIN_RECTANGLE_SIZE_METERS = 0.03f
private const val MAX_RECTANGLE_SIZE_METERS = 5.0f
private const val MIN_RECTANGLE_DEPTH_METERS = 0.1f
private const val MAX_RECTANGLE_DEPTH_METERS = 20.0f
private const val DEFAULT_FALLBACK_DEPTH_METERS = 1.5f
