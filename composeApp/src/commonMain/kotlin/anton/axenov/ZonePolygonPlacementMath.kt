package anton.axenov

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import korlibs.math.geom.Vector3F as Vector3

/**
 * Checks whether the merged zone polygon stays close enough to every source polygon to treat it as placed.
 *
 * @param mergedZone merged zone candidate.
 * @param sourceZones zones that participated in the merge.
 * @return true when the merged polygon remains close to every source polygon.
 */
internal fun wasZoneChangeInsignificant(
    mergedZone: Zone,
    sourceZones: List<Zone>,
): Boolean {
    if (sourceZones.size < 2 || mergedZone.polygonPoints.size < 3)
        return false
    val sourceZone = sourceZones.first()

    if (calculateZonePlaneAngleDegrees(mergedZone, sourceZone) > MERGED_ZONE_MAX_PLANE_ANGLE_DEGREES)
        return false
    val normalizedDifferenceRatio = calculateZonePolygonDifference(
        firstPolygon = mergedZone.polygonPoints,
        secondPolygon = sourceZone.polygonPoints,
        referencePlanePose = mergedZone.planePose,
    ) ?: return false
    return normalizedDifferenceRatio <= MERGED_ZONE_MAX_DIFFERENCE_RATIO
}

/**
 * Calculates the angle between two zone planes in degrees.
 *
 * Plane orientation is direction-agnostic, so opposite normals are treated as the same plane direction.
 *
 * @param firstZone first zone.
 * @param secondZone second zone.
 * @return plane angle in degrees in range `[0, 90]`.
 */
internal fun calculateZonePlaneAngleDegrees(
    firstZone: Zone,
    secondZone: Zone,
): Float {
    val firstNormal = firstZone.planePose.normal.normalized()
    val secondNormal = secondZone.planePose.normal.normalized()
    val dot = abs(firstNormal.dot(secondNormal)).coerceIn(0f, 1f)
    return (acos(dot.toDouble()) * 180.0 / PI).toFloat()
}

/**
 * Calculates polygon difference on a shared plane using a platform-specific geometry engine.
 *
 * @param firstPolygon first world-space polygon.
 * @param secondPolygon second world-space polygon.
 * @param referencePlanePose plane used to project both polygons to 2D.
 * @return difference metrics or null when polygons cannot be compared.
 */
internal expect fun calculateZonePolygonDifference(
    firstPolygon: List<Vector3>,
    secondPolygon: List<Vector3>,
    referencePlanePose: PlanePose,
): Float?

private const val MERGED_ZONE_MAX_DIFFERENCE_RATIO = 0.1f
private const val MERGED_ZONE_MAX_PLANE_ANGLE_DEGREES = 15f
