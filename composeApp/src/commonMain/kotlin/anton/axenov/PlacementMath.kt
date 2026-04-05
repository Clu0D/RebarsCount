package anton.axenov

import kotlin.random.Random
import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3

/**
 * Plane pose payload returned by shared fitting logic.
 *
 * @param center plane center.
 * @param rotation orientation quaternion where local +Z is the plane normal.
 * @param normal fitted normal.
 */
data class PlanePose(
    val center: Vector3,
    val rotation: Quaternion,
    val normal: Vector3,
    val offsetD: Float = -normal.dot(center),
)

/**
 * Result of 3D plane fitting attempt.
 *
 * @param pose fitted plane pose or null when fitting failed.
 * @param depthMeters estimated camera-to-plane-center distance.
 * @param details fitting diagnostics.
 */
data class PlaneFitResult(
    val pose: PlanePose?,
    val depthMeters: Float?,
    val inlierPoints: List<Vector3> = emptyList(),
    val details: String,
)

/**
 * Result of linear plane regression `z = a*x + b*y + c`.
 *
 * @param coefficients regression coefficients where `x = a`, `y = b`, `z = c`.
 * @param inlierPoints points kept after robust residual filtering.
 */
data class PlaneRegressionResult(
    val coefficients: Vector3,
    val inlierPoints: List<Vector3>,
)

/**
 * Pixel point in image coordinates.
 *
 * @param x X coordinate.
 * @param y Y coordinate.
 */
data class ImagePoint(
    val x: Int,
    val y: Int,
)

/**
 * View point in render-surface coordinates.
 *
 * @param xPx X coordinate in pixels.
 * @param yPx Y coordinate in pixels.
 */
data class ViewPoint(
    val xPx: Float,
    val yPx: Float,
)

/**
 * Candidate depth sample used for confidence tie-breaking.
 *
 * @param x depth-map X coordinate.
 * @param y depth-map Y coordinate.
 * @param depthMillimeters sampled depth value in millimeters.
 */
data class DepthCandidate(
    val x: Int,
    val y: Int,
    val depthMillimeters: Int,
)

/**
 * Selects the median point from confidence-tied candidates.
 *
 * Candidates are ordered by `(x, y)` and the middle point is selected.
 *
 * @param candidates tied candidates with equal confidence.
 * @return median point candidate or null when input is empty.
 */
fun selectMedianPointCandidate(candidates: List<DepthCandidate>): DepthCandidate? {
    if (candidates.isEmpty()) {
        return null
    }
    val sorted = candidates.sortedWith(compareBy<DepthCandidate>({ it.x }, { it.y }))
    return sorted[sorted.size / 2]
}

/**
 * Samples random image points inside a bounding box and always includes its center.
 *
 * @param screenBoundingBox in image pixels.
 * @param imageWidth image width.
 * @param imageHeight image height.
 * @param count number of points to sample.
 * @param random random source.
 * @return sampled image-space points.
 */
fun sampleImagePointsInScreenBoundingBox(
    screenBoundingBox: ScreenBoundingBox,
    imageWidth: Int,
    imageHeight: Int,
    count: Int,
    random: Random,
): List<ImagePoint> {
    if (imageWidth <= 0 || imageHeight <= 0) {
        return emptyList()
    }
    val left = screenBoundingBox.left.coerceIn(0, imageWidth - 1)
    val right = screenBoundingBox.right.coerceIn(0, imageWidth - 1).coerceAtLeast(left)
    val top = screenBoundingBox.top.coerceIn(0, imageHeight - 1)
    val bottom = screenBoundingBox.bottom.coerceIn(0, imageHeight - 1).coerceAtLeast(top)

    val points = mutableListOf<ImagePoint>()
    points += ImagePoint((left + right) / 2, (top + bottom) / 2)
    repeat((count - 1).coerceAtLeast(0)) {
        val x = if (left == right) left else random.nextInt(left, right + 1)
        val y = if (top == bottom) top else random.nextInt(top, bottom + 1)
        points += ImagePoint(x, y)
    }
    return points
}

/**
 * Maps image-space points to view-space pixels preserving normalized coordinates.
 *
 * @param imagePoints sampled image points.
 * @param imageWidth image width in pixels.
 * @param imageHeight image height in pixels.
 * @param viewWidth view width in pixels.
 * @param viewHeight view height in pixels.
 * @return mapped view-space points.
 */
fun mapImagePointsToViewPoints(
    imagePoints: List<ImagePoint>,
    imageWidth: Int,
    imageHeight: Int,
    viewWidth: Int,
    viewHeight: Int,
): List<ViewPoint> {
    if (imageWidth <= 0 || imageHeight <= 0 || viewWidth <= 0 || viewHeight <= 0) {
        return emptyList()
    }
    return imagePoints.map { point ->
        ViewPoint(
            xPx = (point.x.toFloat() / imageWidth.toFloat()) * viewWidth.toFloat(),
            yPx = (point.y.toFloat() / imageHeight.toFloat()) * viewHeight.toFloat(),
        )
    }
}

/**
 * Fits a plane and returns pose with +Z aligned to fitted normal.
 *
 * Regression model: `z = a*x + b*y + c`.
 *
 * @param worldPoints world points used for fitting.
 * @param cameraPosition camera position used to orient normal toward camera.
 * @param minPointCount minimum number of points required for fitting.
 * @return plane fit result with diagnostics.
 */
fun fitPlanePoseFromPoints(
    worldPoints: List<Vector3>,
    cameraPosition: Vector3,
    minPointCount: Int,
): PlaneFitResult = if (worldPoints.size < minPointCount) {
    PlaneFitResult(
        pose = null,
        depthMeters = null,
        details = "Need at least $minPointCount points, but got ${worldPoints.size}",
    )
} else try {
    val regressionResult = fitPlaneRegression(worldPoints)
    val (a, b, c) = regressionResult.coefficients
    val fittedPoints = regressionResult.inlierPoints

    // plane normal for z = ax + by + c
    var normal = Vector3(-a, -b, 1f).normalized()

    val center = fittedPoints.reduce { acc, v -> acc + v } / fittedPoints.size.toFloat()

    // flip normal toward camera
    val toCamera = (cameraPosition - center).normalized()
    if (normal.dot(toCamera) < 0f) normal = -normal

    val rotation = Quaternion.fromVectors(Vector3(0f, 0f, 1f), normal)
    val depth = (cameraPosition - center).length

    PlaneFitResult(
        pose = PlanePose(center, rotation, normal),
        depthMeters = depth,
        inlierPoints = fittedPoints,
        details = "ok (inliers=${fittedPoints.size}/${worldPoints.size})",
    )

} catch (e: Exception) {
    PlaneFitResult(
        pose = null,
        depthMeters = null,
        inlierPoints = emptyList(),
        details = e.message ?: "fit failed",
    )
}

expect fun fitPlaneRegression(
    points: List<Vector3>,
): PlaneRegressionResult


/**
 * Translates one detector-space point to image-space coordinates used by placement math.
 *
 * @param x detector-space X coordinate.
 * @param y detector-space Y coordinate.
 * @param width source image width in pixels.
 * @param height source image height in pixels.
 * @param translationVariant detector-to-image translation variant.
 * @return translated and clamped image-space point.
 */
fun translateCoordinates(
    x: Int,
    y: Int,
    width: Int,
    height: Int,
    translationVariant: CoordinateTranslationVariant,
): ImagePoint {
    if (width <= 0 || height <= 0) {
        return ImagePoint(0, 0)
    }

    val safeX = x.coerceIn(0, width - 1)
    val safeY = y.coerceIn(0, height - 1)
    return when (translationVariant) {
        CoordinateTranslationVariant.PORTRAIT -> ImagePoint(
            x = safeY * width / height,
            y = (width - 1 - safeX) * height / width,
        )

        CoordinateTranslationVariant.LANDSCAPE -> ImagePoint(
            x = width - 1 - safeX,
            y = height - 1 - safeY,
        )

        CoordinateTranslationVariant.LANDSCAPE_REVERSED -> ImagePoint(
            x = safeX,
            y = safeY,
        )
    }
}

/**
 * Translation strategy by orientation.
 */
enum class CoordinateTranslationVariant {
    PORTRAIT,
    LANDSCAPE,
    LANDSCAPE_REVERSED,
}
