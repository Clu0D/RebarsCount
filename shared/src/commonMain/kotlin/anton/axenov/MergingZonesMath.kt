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
    if (worldPoints.isEmpty())
        return emptyList()

    val basis = buildPlaneBasis(planePose)

    return worldPoints.map { point ->
        val fromOrigin = point - planePose.center
        HullPoint2d(
            x = fromOrigin.dot(basis.axisX),
            y = fromOrigin.dot(basis.axisY),
            worldPoint = point,
        )
    }.distinctBy { point ->
        "${point.x.roundToHullPrecision()}|${point.y.roundToHullPrecision()}"
    }
}

/**
 * Computes 2D convex hull for polygons known to be on one plane.
 *
 * @param worldPoints projected world-space polygons on a plane.
 * @param planePose plane used for local 2D basis.
 * @return hull vertices in world coordinates.
 */
fun buildConvexHullOnPlane(
    worldPoints: List<List<Vector3F>>,
    planePose: PlanePose,
): List<Vector3F> {
    val flattenedPoints = worldPoints.flatten()
    if (flattenedPoints.isEmpty())
        return flattenedPoints

    val points2d = projectWorldPointsToPlaneCoordinates(
        worldPoints = flattenedPoints,
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
 * Builds an area covered by at least [minConfidence] for multiple polygons known to be on one plane.
 *
 * @param worldPoints projected world-space polygons on a common plane.
 * @param planePose plane used for local 2D basis.
 * @param minConfidence minimum polygon coverage ratio required to keep area.
 * @return polygon in world coordinates.
 */
fun buildConfidenceConvexHullOnPlane(
    worldPoints: List<List<Vector3F>>,
    planePose: PlanePose,
    minConfidence: Float = CONVEX_HULL_CONFIDENCE_RATIO,
): List<Vector3F> {
    val supportedHull = buildConfidenceConvexHullOnPlaneWithGeometry(worldPoints, planePose, minConfidence)
    if (supportedHull != null && supportedHull.size >= 3)
        return supportedHull
    return buildConvexHullOnPlane(
        worldPoints = worldPoints,
        planePose = planePose,
    )
}

/**
 * Builds a confidence-filtered convex hull from geometry-backed polygon overlap calculations.
 *
 * @param worldPoints projected world-space polygons on a common plane.
 * @param planePose plane used for local 2D basis.
 * @param minConfidence minimum polygon coverage ratio required to keep area.
 * @return confidence-filtered convex hull or null when unsupported on the current platform.
 */
internal expect fun buildConfidenceConvexHullOnPlaneWithGeometry(
    worldPoints: List<List<Vector3F>>,
    planePose: PlanePose,
    minConfidence: Float,
): List<Vector3F>?


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

/**
 * One orthonormal 2D basis constructed on top of a plane.
 *
 * @param axisX local X axis on plane.
 * @param axisY local Y axis on plane.
 */
internal data class PlaneBasis2d(
    val axisX: Vector3F,
    val axisY: Vector3F,
)

/**
 * Builds an orthonormal 2D basis for [planePose].
 *
 * @param planePose plane used for local basis.
 * @return local plane basis.
 */
internal fun buildPlaneBasis(planePose: PlanePose): PlaneBasis2d {
    val normal = planePose.normal.normalized()
    val helperAxis = if (kotlin.math.abs(normal.y) < 0.99f) {
        Vector3F(0f, 1f, 0f)
    } else {
        Vector3F(1f, 0f, 0f)
    }
    val axisX = normal.cross(helperAxis).normalized()
    val axisY = normal.cross(axisX).normalized()
    return PlaneBasis2d(
        axisX = axisX,
        axisY = axisY,
    )
}

/**
 * Converts one plane-local 2D point back into world coordinates on [planePose].
 *
 * @param x plane-local X coordinate.
 * @param y plane-local Y coordinate.
 * @param planePose plane used for local basis.
 * @return world-space point on the plane.
 */
internal fun planeCoordinatesToWorldPoint(
    x: Float,
    y: Float,
    planePose: PlanePose,
): Vector3F {
    val basis = buildPlaneBasis(planePose)
    return planePose.center + basis.axisX * x + basis.axisY * y
}

private const val HULL_POINT_PRECISION_SCALE = 100_000f
private const val CONVEX_HULL_CONFIDENCE_RATIO = 0.3f