package anton.axenov

import kotlin.math.max
import korlibs.math.geom.Vector3F as Vector3

/**
 * Calculates polygon difference metrics with JTS on Android.
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

private const val MIN_ZONE_POLYGON_AREA = 1e-6f
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