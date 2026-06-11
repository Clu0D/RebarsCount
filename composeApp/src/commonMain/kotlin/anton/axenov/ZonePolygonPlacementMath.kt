package anton.axenov

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import korlibs.math.geom.Vector3F as Vector3

/**
 * Merge-change evaluation result.
 *
 * @param wasChangeInsignificant true when merged polygon stayed close enough to source polygon.
 * @param planeAngleDegrees angle between merged and source zone planes in degrees.
 * @param normalizedDifferenceRatio normalized polygon symmetric-difference ratio.
 */
internal data class ZoneChangeInsignificanceResult(
    val planeAngleDegrees: Float,
    val normalizedDifferenceRatio: Float,
) {
    val wasChangeInsignificant =
                planeAngleDegrees < MERGED_ZONE_MAX_PLANE_ANGLE_DEGREES &&
                normalizedDifferenceRatio <= MERGED_ZONE_MAX_DIFFERENCE_RATIO
    val mergeLabelText by lazy {
        "\nmerge ang=${planeAngleDegrees.toPrecision(2)}deg, " +
                "diff=${normalizedDifferenceRatio.toPrecision(2)}"
    }
}

/**
 * Checks whether the merged zone polygon stays close enough to every source polygon to treat it as placed.
 *
 * @param mergedZone merged zone candidate.
 * @param sourceZones zones that participated in the merge.
 * @return is result zone close to the original zone with calculated metrics.
 */
internal fun wasZoneChangeInsignificant(
    mergedZone: Zone,
    sourceZones: List<Zone>,
): ZoneChangeInsignificanceResult? {
    if (sourceZones.size < 2 || mergedZone.polygonPoints.size < 3)
        return null

    val sourceZone = sourceZones.first()

    val planeAngleDegrees = calculateZonePlaneAngleDegrees(mergedZone, sourceZone)
    val normalizedDifferenceRatio = calculateZonePolygonDifference(
        firstPolygon = mergedZone.polygonPoints,
        secondPolygon = sourceZone.polygonPoints,
        referencePlanePose = mergedZone.planePose,
    ) ?: return null
    return ZoneChangeInsignificanceResult(planeAngleDegrees, normalizedDifferenceRatio)
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

/**
 * Keeps only those projection inputs whose projected polygons intersect the final merged hull.
 *
 * @param projectionInputs source projection inputs participating in the merge.
 * @param mergedPolygon final merged polygon built on [referencePlanePose].
 * @param referencePlanePose plane used to project polygons into local 2D coordinates.
 * @return surviving projection inputs that remain geometrically consistent with the merged result.
 */
internal expect fun filterProjectionInputsByMergedHull(
    projectionInputs: List<ZoneProjectionInput>,
    mergedPolygon: List<Vector3>,
    referencePlanePose: PlanePose,
): List<ZoneProjectionInput>

private const val MERGED_ZONE_MAX_DIFFERENCE_RATIO = 0.5f
private const val MERGED_ZONE_MAX_PLANE_ANGLE_DEGREES = 25f
