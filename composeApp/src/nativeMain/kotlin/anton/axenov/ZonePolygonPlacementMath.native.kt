package anton.axenov

import korlibs.math.geom.Vector3F as Vector3

/**
 * Calculates polygon difference metrics on native targets.
 *
 * Native targets do not have JTS in this module, so merge-based placement promotion is disabled there.
 *
 * @param firstPolygon first world-space polygon.
 * @param secondPolygon second world-space polygon.
 * @param referencePlanePose plane used to project both polygons to 2D.
 * @return null because this comparison is not supported on native targets yet.
 */
internal actual fun calculateZonePolygonDifference(
    firstPolygon: List<Vector3>,
    secondPolygon: List<Vector3>,
    referencePlanePose: PlanePose,
): Float? = null

/**
 * Native fallback keeps all projection inputs because JTS-backed geometric filtering is unavailable here.
 *
 * @param projectionInputs source projection inputs participating in the merge.
 * @param mergedPolygon final merged polygon.
 * @param referencePlanePose plane used to compare geometries.
 * @return original projection inputs unchanged.
 */
internal actual fun filterProjectionInputsByMergedHull(
    projectionInputs: List<ZoneProjectionInput>,
    mergedPolygon: List<Vector3>,
    referencePlanePose: PlanePose,
): List<ZoneProjectionInput> = projectionInputs
