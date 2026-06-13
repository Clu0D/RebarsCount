package anton.axenov

import kotlin.random.Random
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F

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
    val inlierPoints: List<Vector3F> = emptyList(),
    val details: String,
)

/**
 * Result of linear plane regression `z = a*x + b*y + c`.
 *
 * @param coefficients regression coefficients where `x = a`, `y = b`, `z = c`.
 * @param inlierPoints points kept after robust residual filtering.
 */
data class PlaneRegressionResult(
    val coefficients: Vector3F,
    val inlierPoints: List<Vector3F>,
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
 * Selects the median-depth point from confidence-tied candidates.
 *
 * Candidates are ordered by `(depthMillimeters, x, y)` and the middle point is selected.
 *
 * @param candidates tied candidates with equal confidence.
 * @return median point candidate or null when input is empty.
 */
fun selectMedianPointCandidate(candidates: List<DepthCandidate>): DepthCandidate? {
    if (candidates.isEmpty()) {
        return null
    }
    val sorted = candidates.sortedWith(
        compareBy<DepthCandidate>({ it.depthMillimeters }, { it.x }, { it.y }),
    )
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
 * Samples random image points inside a polygon and always includes one interior anchor point.
 *
 * @param screenPolygon polygon in image pixels.
 * @param imageWidth image width.
 * @param imageHeight image height.
 * @param count number of points to sample.
 * @param random random source.
 * @return sampled image-space points inside the polygon.
 */
fun sampleImagePointsInScreenPolygon(
    screenPolygon: List<ImagePoint>,
    imageWidth: Int,
    imageHeight: Int,
    count: Int,
    random: Random,
): List<ImagePoint> {
    if (imageWidth <= 0 || imageHeight <= 0 || screenPolygon.size < 3) {
        return emptyList()
    }
    val polygon = screenPolygon.map { point ->
        ImagePoint(
            x = point.x.coerceIn(0, imageWidth - 1),
            y = point.y.coerceIn(0, imageHeight - 1),
        )
    }
    val bounds = ScreenBoundingBox(polygon)
    val center = ImagePoint(bounds.centerX(), bounds.centerY())
    val anchor = center.takeIf { point -> isImagePointInsidePolygon(point, polygon) } ?: polygon.first()
    val points = mutableListOf(anchor)
    val targetCount = count.coerceAtLeast(1)
    repeat(POLYGON_SAMPLE_ATTEMPT_MULTIPLIER * targetCount) {
        if (points.size >= targetCount) {
            return points
        }
        val candidate = ImagePoint(
            x = random.nextInt(bounds.left, bounds.right + 1),
            y = random.nextInt(bounds.top, bounds.bottom + 1),
        )
        if (isImagePointInsidePolygon(candidate, polygon)) {
            points += candidate
        }
    }
    return points
}

/**
 * Checks whether an image point lies inside or on the border of a polygon.
 *
 * @param point tested image point.
 * @param polygon polygon vertices.
 * @return true when [point] belongs to the polygon.
 */
internal fun isImagePointInsidePolygon(point: ImagePoint, polygon: List<ImagePoint>): Boolean {
    if (polygon.size < 3) {
        return false
    }
    var inside = false
    var previous = polygon.last()
    polygon.forEach { current ->
        if (isPointOnSegment(point, previous, current)) {
            return true
        }
        val crossesHorizontalRay =
            (current.y > point.y) != (previous.y > point.y) &&
                point.x.toDouble() <
                (previous.x - current.x).toDouble() * (point.y - current.y) /
                (previous.y - current.y).toDouble() + current.x
        if (crossesHorizontalRay) {
            inside = !inside
        }
        previous = current
    }
    return inside
}

/**
 * Checks whether an image point lies on a line segment.
 */
private fun isPointOnSegment(point: ImagePoint, first: ImagePoint, second: ImagePoint): Boolean {
    val cross = (point.y - first.y).toLong() * (second.x - first.x) -
        (point.x - first.x).toLong() * (second.y - first.y)
    if (cross != 0L) {
        return false
    }
    return point.x in minOf(first.x, second.x)..maxOf(first.x, second.x) &&
        point.y in minOf(first.y, second.y)..maxOf(first.y, second.y)
}

private const val POLYGON_SAMPLE_ATTEMPT_MULTIPLIER = 100

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
    worldPoints: List<Vector3F>,
    cameraPosition: Vector3F,
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
    var normal = Vector3F(-a, -b, 1f).normalized()

    val center = fittedPoints.reduce { acc, v -> acc + v } / fittedPoints.size.toFloat()

    // flip normal toward camera
    val toCamera = (cameraPosition - center).normalized()
    if (normal.dot(toCamera) < 0f) normal = -normal

    val rotation = Quaternion.fromVectors(Vector3F(0f, 0f, 1f), normal)
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
    points: List<Vector3F>,
): PlaneRegressionResult


