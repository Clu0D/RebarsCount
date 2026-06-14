package anton.axenov

import kotlin.math.max
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import korlibs.math.geom.Vector3F as Vector3

private val zonePolygonGeometryFactory = GeometryFactory()

/**
 * Calculates polygon difference metrics with JTS on JVM.
 *
 * 3D polygons are projected to the same plane-local 2D coordinate system before comparison.
 *
 * @param firstPolygon first world-space polygon.
 * @param secondPolygon second world-space polygon.
 * @param referencePlanePose plane used to project both polygons to 2D.
 * @return difference metrics or null when polygons cannot be compared.
 */
internal actual fun calculateZonePolygonDifference(
    firstPolygon: List<Vector3>,
    secondPolygon: List<Vector3>,
    referencePlanePose: PlanePose,
): Float? {
    val firstGeometry = createPlanePolygonGeometry(firstPolygon, referencePlanePose) ?: return null
    val secondGeometry = createPlanePolygonGeometry(secondPolygon, referencePlanePose) ?: return null
    val firstArea = firstGeometry.area.toFloat()
    val secondArea = secondGeometry.area.toFloat()
    if (firstArea <= MIN_ZONE_POLYGON_AREA || secondArea <= MIN_ZONE_POLYGON_AREA) {
        return null
    }
    val symmetricDifferenceArea = firstGeometry.symDifference(secondGeometry).area.toFloat()
    val denominator = max(firstArea, secondArea).coerceAtLeast(MIN_ZONE_POLYGON_AREA)
    return symmetricDifferenceArea / denominator
}

/**
 * Filters projection inputs by intersection with the final merged hull using JTS.
 *
 * @param projectionInputs source projection inputs participating in the merge.
 * @param mergedPolygon final merged polygon.
 * @param referencePlanePose plane used to compare geometries.
 * @return projection inputs whose projected polygons intersect the merged hull.
 */
internal actual fun filterProjectionInputsByMergedHull(
    projectionInputs: List<ZoneProjectionInput>,
    mergedPolygon: List<Vector3>,
    referencePlanePose: PlanePose,
): List<ZoneProjectionInput> {
    val mergedGeometry = createPlanePolygonGeometry(mergedPolygon, referencePlanePose)
        ?: return emptyList()
    return projectionInputs.filter { input ->
        val inputPolygon = input.projectToPlane(referencePlanePose) ?: return@filter false
        val inputGeometry = createPlanePolygonGeometry(inputPolygon, referencePlanePose) ?: return@filter false
        !inputGeometry.intersection(mergedGeometry).isEmpty
    }
}

/**
 * Builds a normalized JTS polygon from a world-space polygon.
 *
 * @param polygonPoints polygon points in world coordinates.
 * @param planePose reference plane used to project points to 2D.
 * @return normalized polygon geometry or null when input is degenerate.
 */
private fun createPlanePolygonGeometry(
    polygonPoints: List<Vector3>,
    planePose: PlanePose,
): Geometry? {
    val projectedPoints = projectWorldPointsToPlaneCoordinates(polygonPoints, planePose)
    if (projectedPoints.size < 3) {
        return null
    }
    val coordinates = Array(projectedPoints.size + 1) { index ->
        val point = if (index == projectedPoints.size) projectedPoints.first() else projectedPoints[index]
        Coordinate(point.x.toDouble(), point.y.toDouble())
    }
    val polygon = zonePolygonGeometryFactory.createPolygon(coordinates)
    val normalizedPolygon = polygon.buffer(0.0)
    return if (normalizedPolygon.isEmpty) null else normalizedPolygon
}

private const val MIN_ZONE_POLYGON_AREA = 1e-6f
