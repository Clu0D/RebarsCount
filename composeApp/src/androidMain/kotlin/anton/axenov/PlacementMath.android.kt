package anton.axenov

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import korlibs.math.geom.Vector3F as Vector3

/**
 * Fits robust linear plane regression `z = a*x + b*y + c` for Android target.
 *
 * Iteratively:
 * 1) fits using all active points,
 * 2) computes absolute residuals,
 * 3) keeps the smallest-residual subset (70%),
 * 4) refits for 3 rounds.
 *
 * @param points input world-space points.
 * @return regression coefficients and inlier points kept by trimming.
 */
actual fun fitPlaneRegression(points: List<Vector3>): PlaneRegressionResult {
    require(points.size >= MIN_ROBUST_POINT_COUNT) {
        "Need at least $MIN_ROBUST_POINT_COUNT points, but got ${points.size}"
    }

    var activePoints = points
    var coefficients = estimateCoefficients(activePoints)
    repeat(ROBUST_REFIT_ITERATIONS) {
        if (activePoints.size <= MIN_ROBUST_POINT_COUNT) return@repeat
        val residualsByPoint = activePoints.map { point ->
            val predictedZ = coefficients.x * point.x + coefficients.y * point.y + coefficients.z
            point to kotlin.math.abs(point.z - predictedZ)
        }.sortedBy { (_, residual) -> residual }
        val pointsToKeep = (activePoints.size * INLIER_KEEP_RATIO).toInt()
            .coerceIn(MIN_ROBUST_POINT_COUNT, activePoints.size)
        activePoints = residualsByPoint.take(pointsToKeep).map { (point, _) -> point }
        coefficients = estimateCoefficients(activePoints)
    }

    return PlaneRegressionResult(
        coefficients = coefficients,
        inlierPoints = activePoints,
    )
}

private fun estimateCoefficients(points: List<Vector3>): Vector3 {
    val features = Array(points.size) { i ->
        doubleArrayOf(
            points[i].x.toDouble(),
            points[i].y.toDouble()
        )
    }
    val target = DoubleArray(points.size) { i ->
        points[i].z.toDouble()
    }

    val regression = OLSMultipleLinearRegression()
    regression.newSampleData(target, features)
    val coefficients = regression.estimateRegressionParameters()

    val c = coefficients[0]
    val a = coefficients[1]
    val b = coefficients[2]

    return Vector3(a.toFloat(), b.toFloat(), c.toFloat())
}

private const val INLIER_KEEP_RATIO = 0.7f
private const val ROBUST_REFIT_ITERATIONS = 3
private const val MIN_ROBUST_POINT_COUNT = 3
