package anton.axenov

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * StarDist polygon candidate.
 *
 * @param center polygon center in model input coordinates.
 * @param score probability score.
 * @param distances radial distances for all rays.
 */
internal data class StarDistPolygonCandidate(
    val center: FloatImagePoint,
    val score: Float,
    val distances: FloatArray,
)

/**
 * Retained StarDist polygon after NMS.
 *
 * @param center polygon center in model input coordinates.
 * @param score probability score.
 * @param vertices polygon vertices in model input coordinates.
 */
internal data class StarDistPolygon(
    val center: FloatImagePoint,
    val score: Float,
    val vertices: List<FloatImagePoint>,
)

/**
 * Common lightweight StarDist NMS implementation.
 */
internal object StarDistPolygonNms {
    /**
     * Reconstructs polygons from dense probability and radial-distance tensors.
     *
     * @param probability flattened probability map.
     * @param distances flattened distance tensor.
     * @param height probability-map height.
     * @param width probability-map width.
     * @param rays number of rays per pixel.
     * @param probabilityThreshold minimal probability used for candidate selection.
     * @param nmsThreshold maximal allowed normalized overlap.
     * @param gridY vertical grid factor between probability map and input image.
     * @param gridX horizontal grid factor between probability map and input image.
     * @param border ignored border width in probability-map pixels.
     * @param maxCandidates maximal number of sorted candidates to process.
     * @return retained polygons sorted by descending score.
     */
    fun fromDensePrediction(
        probability: FloatArray,
        distances: FloatArray,
        height: Int,
        width: Int,
        rays: Int,
        probabilityThreshold: Float,
        nmsThreshold: Float,
        gridY: Int,
        gridX: Int,
        border: Int,
        maxCandidates: Int,
    ): List<StarDistPolygon> {
        require(probability.size == height * width) {
            "probability size must match height * width"
        }
        require(distances.size == height * width * rays) {
            "distance size must match height * width * rays"
        }
        val candidates = mutableListOf<StarDistPolygonCandidate>()
        for (y in border until (height - border).coerceAtLeast(border)) {
            for (x in border until (width - border).coerceAtLeast(border)) {
                val pixelIndex = y * width + x
                val score = probability[pixelIndex]
                if (score < probabilityThreshold) {
                    continue
                }
                candidates += StarDistPolygonCandidate(
                    center = FloatImagePoint(
                        x = x * gridX.toFloat(),
                        y = y * gridY.toFloat(),
                    ),
                    score = score,
                    distances = distances.copyOfRange(pixelIndex * rays, (pixelIndex + 1) * rays),
                )
            }
        }
        return suppress(
            candidates = candidates,
            nmsThreshold = nmsThreshold,
            maxCandidates = maxCandidates,
        )
    }

    /**
     * Runs sparse StarDist NMS on already reconstructed candidates.
     *
     * @param candidates sparse polygon candidates.
     * @param nmsThreshold maximal allowed normalized overlap.
     * @param maxCandidates maximal number of sorted candidates to process.
     * @return retained polygons sorted by descending score.
     */
    fun suppress(
        candidates: List<StarDistPolygonCandidate>,
        nmsThreshold: Float,
        maxCandidates: Int,
    ): List<StarDistPolygon> {
        val sorted = candidates
            .sortedByDescending { candidate -> candidate.score }
            .take(maxCandidates)
            .map { candidate ->
                StarDistPolygon(
                    center = candidate.center,
                    score = candidate.score,
                    vertices = reconstructStarVertices(candidate.center, candidate.distances),
                )
            }
        val kept = mutableListOf<StarDistPolygon>()
        sorted.forEach { polygon ->
            val shouldKeep = kept.none { accepted ->
                normalizedPolygonOverlap(accepted.vertices, polygon.vertices) > nmsThreshold
            }
            if (shouldKeep) {
                kept += polygon
            }
        }
        return kept
    }

    /**
     * Reconstructs one polygon from its center and radial distances.
     *
     * @param center polygon center in image coordinates.
     * @param distances radial distances for all rays.
     * @return polygon vertices in counter-clockwise angle order.
     */
    fun reconstructStarVertices(
        center: FloatImagePoint,
        distances: FloatArray,
    ): List<FloatImagePoint> {
        require(distances.size >= 3) {
            "StarDist polygon requires at least three rays"
        }
        return distances.indices.map { rayIndex ->
            val angle = 2.0 * PI * rayIndex / distances.size
            FloatImagePoint(
                x = center.x + distances[rayIndex] * cos(angle).toFloat(),
                y = center.y + distances[rayIndex] * sin(angle).toFloat(),
            )
        }
    }
}

/**
 * Returns one safe grid ratio between the ONNX input and dense StarDist output.
 *
 * @param inputSize input dimension in pixels.
 * @param outputSize output dimension in pixels.
 * @return positive grid step.
 */
internal fun inferStarDistGrid(inputSize: Int, outputSize: Int): Int {
    return max(1, if (outputSize > 0) inputSize / outputSize else 1)
}
