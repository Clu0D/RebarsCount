package anton.axenov

import korlibs.math.geom.Vector3
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.operation.polygonize.Polygonizer
import org.locationtech.jts.operation.union.UnaryUnionOp
import korlibs.math.geom.Vector3F
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.GeometryFactory
import kotlin.math.ceil

private val zonePolygonGeometryFactory = GeometryFactory()

/**
 * Builds a confidence-filtered convex hull by finding points that are covered by enough polygons.
 *
 * @param worldPoints projected world-space polygons on a common plane.
 * @param planePose plane used for local 2D basis.
 * @param minConfidence minimum polygon coverage ratio required to keep area.
 * @return confidence-filtered convex hull or null when supported area is degenerate.
 */
internal actual fun buildConfidenceConvexHullOnPlaneWithGeometry(
    worldPoints: List<List<Vector3F>>,
    planePose: PlanePose,
    minConfidence: Float,
): List<Vector3F>? {
    val polygons = worldPoints.mapNotNull { polygon ->
        createPlanePolygonGeometry(polygon, planePose)
    }
    if (polygons.isEmpty())
        return null

    val minSupportCount = ceil(polygons.size * minConfidence).toInt().coerceAtLeast(1)
    val nodedBoundary = UnaryUnionOp.union(polygons.map { polygon -> polygon.boundary })
    val polygonizer = Polygonizer()
    polygonizer.add(nodedBoundary)
    val faces = polygonizer.polygons
        .filterIsInstance<Geometry>()
        .filter { face -> !face.isEmpty && face.area > MIN_CONFIDENCE_POLYGON_AREA }
        .filter { face ->
            polygons.count { polygon -> polygon.covers(face.interiorPoint) } >= minSupportCount
        }
    if (faces.isEmpty())
        return null

    val confidenceHull = UnaryUnionOp.union(faces).convexHull()
    if (confidenceHull.isEmpty || confidenceHull.coordinates.size < 3)
        return null

    return confidenceHull.coordinates
        .dropLast(1)
        .map { coordinate ->
            planeCoordinatesToWorldPoint(
                x = coordinate.x.toFloat(),
                y = coordinate.y.toFloat(),
                planePose = planePose,
            )
        }
}


/**
 * Builds a normalized JTS polygon from a world-space polygon.
 *
 * @param polygonPoints polygon points in world coordinates.
 * @param planePose reference plane used to project points to 2D.
 * @return normalized polygon geometry or null when input is degenerate.
 */
fun createPlanePolygonGeometry(
    polygonPoints: List<Vector3>,
    planePose: PlanePose,
): Geometry? {
    val projectedPoints = projectWorldPointsToPlaneCoordinates(polygonPoints, planePose)
    if (projectedPoints.size < 3)
        return null

    val coordinates = Array(projectedPoints.size + 1) { index ->
        val point = if (index == projectedPoints.size) projectedPoints.first() else projectedPoints[index]
        Coordinate(point.x.toDouble(), point.y.toDouble())
    }
    val polygon = zonePolygonGeometryFactory.createPolygon(coordinates)
    val normalizedPolygon = polygon.buffer(0.0)
    return if (normalizedPolygon.isEmpty) null else normalizedPolygon
}

private const val MIN_CONFIDENCE_POLYGON_AREA = 1e-6
