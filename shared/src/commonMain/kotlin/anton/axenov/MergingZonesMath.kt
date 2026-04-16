package anton.axenov

import korlibs.math.geom.Vector3F
import kotlin.math.roundToInt

/**
 * Projects world-space points to a 2D coordinate system local to [planePose].
 *
 * @param worldPoints points on a common plane.
 * @param planePose plane used for local 2D basis.
 * @return plane-local points.
 */
fun projectWorldPointsToPlaneCoordinates(
    worldPoints: List<Vector3F>,
    planePose: PlanePose,
): List<HullPoint2d> {
    if (worldPoints.size <= 1) {
        return emptyList()
    }
    val normal = planePose.normal.normalized()
    val helperAxis = if (kotlin.math.abs(normal.y) < 0.99f) {
        Vector3F(0f, 1f, 0f)
    } else {
        Vector3F(1f, 0f, 0f)
    }
    val axisX = normal.cross(helperAxis).normalized()
    val axisY = normal.cross(axisX).normalized()

    return worldPoints.map { point ->
        val fromOrigin = point - planePose.center
        HullPoint2d(
            x = fromOrigin.dot(axisX),
            y = fromOrigin.dot(axisY),
            worldPoint = point,
        )
    }.distinctBy { point ->
        "${point.x.roundToHullPrecision()}|${point.y.roundToHullPrecision()}"
    }
}

/**
 * Computes 2D convex hull for world-space points known to be on one plane.
 *
 * @param worldPoints points on a common plane.
 * @param planePose plane used for local 2D basis.
 * @return hull vertices in world coordinates.
 */
fun buildConvexHullOnPlane(
    worldPoints: List<Vector3F>,
    planePose: PlanePose,
): List<Vector3F> {
    if (worldPoints.size <= 1) {
        return worldPoints
    }
    val points2d = projectWorldPointsToPlaneCoordinates(
        worldPoints = worldPoints,
        planePose = planePose,
    )
    if (points2d.size <= 2) {
        return points2d.map { it.worldPoint }
    }

    val sorted = points2d.sortedWith(compareBy<HullPoint2d>({ it.x }, { it.y }))
    val lower = mutableListOf<HullPoint2d>()
    sorted.forEach { point ->
        while (lower.size >= 2 && cross(lower[lower.size - 2], lower[lower.size - 1], point) <= 0f) {
            lower.removeAt(lower.lastIndex)
        }
        lower += point
    }

    val upper = mutableListOf<HullPoint2d>()
    sorted.asReversed().forEach { point ->
        while (upper.size >= 2 && cross(upper[upper.size - 2], upper[upper.size - 1], point) <= 0f) {
            upper.removeAt(upper.lastIndex)
        }
        upper += point
    }

    if (lower.isNotEmpty()) {
        lower.removeAt(lower.lastIndex)
    }
    if (upper.isNotEmpty()) {
        upper.removeAt(upper.lastIndex)
    }
    return (lower + upper).map { it.worldPoint }
}

/**
 * Computes oriented area sign for 2D turn `(a -> b -> c)`.
 *
 * @param a first point.
 * @param b second point.
 * @param c third point.
 * @return positive for left turn, negative for right turn, zero for collinear.
 */
private fun cross(a: HullPoint2d, b: HullPoint2d, c: HullPoint2d): Float {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/**
 * Rounds float value to stabilize deduplication of nearly identical hull points.
 *
 * @return rounded value scaled by [HULL_POINT_PRECISION_SCALE].
 */
private fun Float.roundToHullPrecision(): Int {
    return (this * HULL_POINT_PRECISION_SCALE).roundToInt()
}

/**
 * One 2D hull point with world-space source.
 *
 * @param x plane-local X coordinate.
 * @param y plane-local Y coordinate.
 * @param worldPoint original world-space point.
 */
data class HullPoint2d(
    val x: Float,
    val y: Float,
    val worldPoint: Vector3F,
)

private const val HULL_POINT_PRECISION_SCALE = 100_000f