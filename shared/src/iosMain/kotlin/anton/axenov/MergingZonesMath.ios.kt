package anton.axenov

import korlibs.math.geom.Vector3F

/**
 * iOS fallback for confidence-filtered convex hulls.
 *
 * JTS is not available on this target in the shared module yet, so callers fall back to the
 * existing flat-point convex hull implementation.
 *
 * @param worldPoints projected world-space polygons on a common plane.
 * @param planePose plane used for local 2D basis.
 * @param minConfidence minimum polygon coverage ratio required to keep area.
 * @return null to trigger common fallback logic.
 */
internal actual fun buildConfidenceConvexHullOnPlaneWithGeometry(
    worldPoints: List<List<Vector3F>>,
    planePose: PlanePose,
    minConfidence: Float,
): List<Vector3F>? = null
