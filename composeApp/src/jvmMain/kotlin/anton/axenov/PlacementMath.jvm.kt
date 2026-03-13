package anton.axenov

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import korlibs.math.geom.Vector3F as Vector3

/**
 * Fits linear plane regression `z = a*x + b*y + c` for JVM desktop target.
 *
 * @param points input world-space points.
 * @return vector where `x = a`, `y = b`, `z = c`.
 */
actual fun fitPlaneRegression(points: List<Vector3>): Vector3 {
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
