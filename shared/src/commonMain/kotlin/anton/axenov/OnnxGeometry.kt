package anton.axenov

import kotlin.math.abs

/**
 * Returns the convex hull of one point cloud using the monotonic chain algorithm.
 *
 * @param points unordered input points.
 * @return hull vertices in counter-clockwise order without repeating the first point.
 */
internal fun convexHull(points: List<FloatImagePoint>): List<FloatImagePoint> {
    val uniquePoints = points.distinctBy { point -> point.x to point.y }
        .sortedWith(compareBy<FloatImagePoint> { point -> point.x }.thenBy { point -> point.y })
    if (uniquePoints.size <= 2) {
        return uniquePoints
    }

    val lower = mutableListOf<FloatImagePoint>()
    uniquePoints.forEach { point ->
        while (lower.size >= 2 && cross(lower[lower.lastIndex - 1], lower.last(), point) <= 0f) {
            lower.removeAt(lower.lastIndex)
        }
        lower += point
    }

    val upper = mutableListOf<FloatImagePoint>()
    uniquePoints.asReversed().forEach { point ->
        while (upper.size >= 2 && cross(upper[upper.lastIndex - 1], upper.last(), point) <= 0f) {
            upper.removeAt(upper.lastIndex)
        }
        upper += point
    }

    return (lower.dropLast(1) + upper.dropLast(1)).distinct()
}

/**
 * Returns the signed triangle cross product for three points.
 *
 * @param origin first point.
 * @param middle second point.
 * @param target third point.
 * @return positive value for a left turn, negative for a right turn.
 */
internal fun cross(
    origin: FloatImagePoint,
    middle: FloatImagePoint,
    target: FloatImagePoint,
): Float {
    return (middle.x - origin.x) * (target.y - origin.y) -
        (middle.y - origin.y) * (target.x - origin.x)
}

/**
 * Returns the area of one polygon.
 *
 * @param polygon polygon vertices in order.
 * @return non-negative polygon area.
 */
internal fun polygonArea(polygon: List<FloatImagePoint>): Float {
    if (polygon.size < 3) {
        return 0f
    }
    var doubledArea = 0f
    polygon.indices.forEach { index ->
        val current = polygon[index]
        val next = polygon[(index + 1) % polygon.size]
        doubledArea += current.x * next.y - next.x * current.y
    }
    return abs(doubledArea) * 0.5f
}

/**
 * Clips one polygon by another convex polygon using Sutherland-Hodgman clipping.
 *
 * @param subject polygon to clip.
 * @param clip convex clip polygon.
 * @return clipped polygon vertices.
 */
internal fun clipPolygon(
    subject: List<FloatImagePoint>,
    clip: List<FloatImagePoint>,
): List<FloatImagePoint> {
    if (subject.size < 3 || clip.size < 3) {
        return emptyList()
    }
    var output: List<FloatImagePoint> = subject
    clip.indices.forEach { index ->
        val clipStart = clip[index]
        val clipEnd = clip[(index + 1) % clip.size]
        if (output.isEmpty()) {
            return emptyList()
        }
        val input = output
        val nextOutput = mutableListOf<FloatImagePoint>()
        var previous = input.last()
        input.forEach { current ->
            val currentInside = isInsideHalfPlane(current, clipStart, clipEnd)
            val previousInside = isInsideHalfPlane(previous, clipStart, clipEnd)
            if (currentInside) {
                if (!previousInside) {
                    lineIntersection(previous, current, clipStart, clipEnd)?.also { point ->
                        nextOutput.add(point)
                    }
                }
                nextOutput += current
            } else if (previousInside) {
                lineIntersection(previous, current, clipStart, clipEnd)?.also { point ->
                    nextOutput.add(point)
                }
            }
            previous = current
        }
        output = nextOutput
    }
    return output
}

/**
 * Returns the overlap used by StarDist NMS
 *
 * @param first first polygon.
 * @param second second polygon.
 * @return `intersection / min(areaA, areaB)` in range `[0, 1]`.
 */
internal fun normalizedPolygonOverlap(
    first: List<FloatImagePoint>,
    second: List<FloatImagePoint>,
): Float {
    val firstHull = convexHull(first)
    val secondHull = convexHull(second)
    val firstArea = polygonArea(firstHull)
    val secondArea = polygonArea(secondHull)
    if (firstArea <= 0f || secondArea <= 0f) {
        return 0f
    }
    val intersection = polygonArea(clipPolygon(firstHull, secondHull))
    return (intersection / minOf(firstArea, secondArea)).coerceIn(0f, 1f)
}

/**
 * Builds one polygon from a dense foreground mask using a convex hull.
 *
 * @param points foreground samples already mapped into image space.
 * @param fallbackBox fallback rectangle used when the mask is too small.
 * @return polygon with at least four rectangle corners when the hull degenerates.
 */
internal fun polygonFromForegroundPoints(
    points: List<FloatImagePoint>,
    fallbackBox: FloatBoundingBox,
): List<FloatImagePoint> {
    val hull = convexHull(points)
    if (hull.size >= 3) {
        return hull
    }
    return listOf(
        FloatImagePoint(fallbackBox.left, fallbackBox.top),
        FloatImagePoint(fallbackBox.right, fallbackBox.top),
        FloatImagePoint(fallbackBox.right, fallbackBox.bottom),
        FloatImagePoint(fallbackBox.left, fallbackBox.bottom),
    )
}

/**
 * Computes the intersection point of two infinite lines.
 *
 * @param firstStart first line start.
 * @param firstEnd first line end.
 * @param secondStart second line start.
 * @param secondEnd second line end.
 * @return intersection point or null when the lines are parallel.
 */
internal fun lineIntersection(
    firstStart: FloatImagePoint,
    firstEnd: FloatImagePoint,
    secondStart: FloatImagePoint,
    secondEnd: FloatImagePoint,
): FloatImagePoint? {
    val a1 = firstEnd.y - firstStart.y
    val b1 = firstStart.x - firstEnd.x
    val c1 = a1 * firstStart.x + b1 * firstStart.y
    val a2 = secondEnd.y - secondStart.y
    val b2 = secondStart.x - secondEnd.x
    val c2 = a2 * secondStart.x + b2 * secondStart.y
    val determinant = a1 * b2 - a2 * b1
    if (abs(determinant) <= 1e-6f) {
        return null
    }
    return FloatImagePoint(
        x = (b2 * c1 - b1 * c2) / determinant,
        y = (a1 * c2 - a2 * c1) / determinant,
    )
}

/**
 * Checks whether a point lies inside the left half-plane of one directed edge.
 *
 * @param point candidate point.
 * @param edgeStart directed edge start.
 * @param edgeEnd directed edge end.
 * @return true when the point is inside or on the edge.
 */
internal fun isInsideHalfPlane(
    point: FloatImagePoint,
    edgeStart: FloatImagePoint,
    edgeEnd: FloatImagePoint,
): Boolean {
    return cross(edgeStart, edgeEnd, point) >= 0f
}
