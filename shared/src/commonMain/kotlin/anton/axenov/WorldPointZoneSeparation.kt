package anton.axenov

import kotlin.math.exp
import korlibs.math.geom.Vector3F

/**
 * One world point with the final zone assignment produced by post-processing.
 *
 * @param zoneId identifier of the assigned zone.
 * @param worldPoint original reconstructed world point.
 * @param isAnchor true when the point was initially anchored to the zone before propagation.
 */
data class AssignedWorldPoint(
    val zoneId: Long,
    val worldPoint: WorldPoint,
    val isAnchor: Boolean,
)

/**
 * Assigns every reconstructed [WorldPoint] to one concrete [Zone] using a weighted label propagation graph.
 *
 * Anchor points are selected first. A point becomes an anchor for the nearest zone when its distance
 * to that zone center is less than half of the distance to every other zone center.
 * After that the remaining labels are propagated through a k-nearest-neighbour graph.
 *
 * @param worldPoints all reconstructed world points from the current session.
 * @param zones all known zones from the current session.
 * @param neighbourCount number of nearest neighbours used to build the propagation graph.
 * @param maxIterations maximal number of propagation iterations.
 * @return final zone assignment for every input point.
 */
fun assignWorldPointsToZones(
    worldPoints: List<WorldPoint>,
    zones: List<Zone>,
    neighbourCount: Int = DEFAULT_LABEL_PROPAGATION_NEIGHBOUR_COUNT,
    maxIterations: Int = DEFAULT_LABEL_PROPAGATION_MAX_ITERATIONS,
): List<AssignedWorldPoint> {
    if (worldPoints.isEmpty() || zones.isEmpty()) {
        return emptyList()
    }
    if (zones.size == 1) {
        val onlyZoneId = zones.single().id
        return worldPoints.map { worldPoint ->
            AssignedWorldPoint(
                zoneId = onlyZoneId,
                worldPoint = worldPoint,
                isAnchor = true,
            )
        }
    }

    val anchorsByPointIndex = findAnchorZoneIds(worldPoints, zones)
    val adjacencyByPointIndex = buildAdjacencyGraph(worldPoints, neighbourCount)
    val labelsByPointIndex = anchorsByPointIndex.toMutableList()
    val anchorFlagsByPointIndex = anchorsByPointIndex.map { zoneId -> zoneId != null }

    var iteration = 0
    while (iteration < maxIterations) {
        var changed = false
        worldPoints.indices.forEach { pointIndex ->
            if (anchorFlagsByPointIndex[pointIndex]) {
                return@forEach
            }
            val propagatedLabel = selectPropagatedZoneId(
                worldPoint = worldPoints[pointIndex],
                zones = zones,
                neighbours = adjacencyByPointIndex[pointIndex],
                currentLabels = labelsByPointIndex,
            ) ?: return@forEach
            if (labelsByPointIndex[pointIndex] != propagatedLabel) {
                labelsByPointIndex[pointIndex] = propagatedLabel
                changed = true
            }
        }
        if (!changed) {
            break
        }
        iteration++
    }

    return worldPoints.mapIndexed { index, worldPoint ->
        AssignedWorldPoint(
            zoneId = labelsByPointIndex[index] ?: nearestZoneId(worldPoint.position, zones),
            worldPoint = worldPoint,
            isAnchor = anchorFlagsByPointIndex[index],
        )
    }
}

/**
 * Finds initial zone anchors using the "twice closer to one center" rule.
 *
 * @param worldPoints points that should be checked.
 * @param zones all zones competing for assignment.
 * @return zone id for each anchored point or null.
 */
private fun findAnchorZoneIds(
    worldPoints: List<WorldPoint>,
    zones: List<Zone>,
): List<Long?> {
    return worldPoints.map { worldPoint ->
        val distancesByZone = zones
            .map { zone -> zone to squaredDistance(worldPoint.position, zone.center) }
            .sortedBy { (_, distanceSquared) -> distanceSquared }
        val nearest = distancesByZone.firstOrNull() ?: return@map null
        val secondNearest = distancesByZone.getOrNull(1) ?: return@map nearest.first.id
        if (nearest.second * 4f < secondNearest.second) {
            nearest.first.id
        } else {
            null
        }
    }
}

/**
 * Builds one undirected weighted k-nearest-neighbour graph for label propagation.
 *
 * @param worldPoints all graph vertices.
 * @param neighbourCount number of nearest neighbours for every vertex.
 * @return adjacency list with propagation weights.
 */
private fun buildAdjacencyGraph(
    worldPoints: List<WorldPoint>,
    neighbourCount: Int,
): List<List<WeightedNeighbour>> {
    val edgeWeightByVertices = mutableMapOf<Pair<Int, Int>, Float>()
    val boundedNeighbourCount = neighbourCount.coerceAtLeast(1)
        .coerceAtMost((worldPoints.size - 1).coerceAtLeast(1))
    val neighbourDistances = mutableListOf<Float>()

    worldPoints.indices.forEach { sourceIndex ->
        worldPoints.indices
            .asSequence()
            .filter { targetIndex -> targetIndex != sourceIndex }
            .map { targetIndex ->
                IndexedDistance(
                    index = targetIndex,
                    distanceSquared = squaredDistance(
                        first = worldPoints[sourceIndex].position,
                        second = worldPoints[targetIndex].position,
                    ),
                )
            }
            .sortedBy { indexedDistance -> indexedDistance.distanceSquared }
            .take(boundedNeighbourCount)
            .forEach { neighbour ->
                val distance = kotlin.math.sqrt(neighbour.distanceSquared.toDouble()).toFloat()
                neighbourDistances += distance
                edgeWeightByVertices[orderedEdgeKey(sourceIndex, neighbour.index)] = distance
            }
    }

    val sigma = median(neighbourDistances).coerceAtLeast(MIN_LABEL_PROPAGATION_SIGMA)
    val sigmaSquared = sigma * sigma
    val adjacency = List(worldPoints.size) { mutableMapOf<Int, Float>() }
    edgeWeightByVertices.forEach { (edge, distance) ->
        val firstIndex = edge.first
        val secondIndex = edge.second
        val firstPoint = worldPoints[firstIndex]
        val secondPoint = worldPoints[secondIndex]
        val weight = exp(
            -((distance * distance) / sigmaSquared).toDouble(),
        ).toFloat() * firstPoint.confidence * secondPoint.confidence
        adjacency[firstIndex][secondIndex] = weight
        adjacency[secondIndex][firstIndex] = weight
    }
    return adjacency.map { neighbours ->
        neighbours.entries
            .map { (index, weight) -> WeightedNeighbour(index, weight) }
            .sortedByDescending { neighbour -> neighbour.weight }
    }
}

/**
 * Selects the strongest propagated zone label for one vertex from already labelled neighbours.
 *
 * @param worldPoint point being updated.
 * @param zones all available zones used for deterministic tie-breaking.
 * @param neighbours weighted neighbouring vertices.
 * @param currentLabels current labels for all vertices.
 * @return best propagated zone id or null when no labelled neighbours exist yet.
 */
private fun selectPropagatedZoneId(
    worldPoint: WorldPoint,
    zones: List<Zone>,
    neighbours: List<WeightedNeighbour>,
    currentLabels: List<Long?>,
): Long? {
    val scoreByZoneId = mutableMapOf<Long, Float>()
    neighbours.forEach { neighbour ->
        val neighbourZoneId = currentLabels[neighbour.index] ?: return@forEach
        scoreByZoneId[neighbourZoneId] = (scoreByZoneId[neighbourZoneId] ?: 0f) + neighbour.weight
    }
    if (scoreByZoneId.isEmpty()) {
        return null
    }
    val nearestZoneId = nearestZoneId(worldPoint.position, zones)
    return scoreByZoneId.entries
        .sortedWith(
            compareByDescending<Map.Entry<Long, Float>> { entry -> entry.value }
                .thenBy { entry ->
                    if (entry.key == nearestZoneId) {
                        0
                    } else {
                        1
                    }
                }
                .thenBy { entry -> entry.key },
        )
        .first()
        .key
}

/**
 * Returns the nearest zone center for one point.
 *
 * @param point point that should be assigned.
 * @param zones candidate zones.
 * @return identifier of the nearest zone.
 */
private fun nearestZoneId(
    point: Vector3F,
    zones: List<Zone>,
): Long {
    return zones.minBy { zone -> squaredDistance(point, zone.center) }.id
}

/**
 * Computes squared Euclidean distance between two world-space points.
 *
 * @param first first position.
 * @param second second position.
 * @return squared distance.
 */
private fun squaredDistance(
    first: Vector3F,
    second: Vector3F,
): Float {
    val delta = first - second
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
}

/**
 * Returns one ordered undirected edge key.
 *
 * @param first first vertex index.
 * @param second second vertex index.
 * @return stable undirected edge key.
 */
private fun orderedEdgeKey(
    first: Int,
    second: Int,
): Pair<Int, Int> {
    return if (first <= second) {
        first to second
    } else {
        second to first
    }
}

/**
 * Returns the median value or a safe fallback for an empty list.
 *
 * @param values input values.
 * @return median or [MIN_LABEL_PROPAGATION_SIGMA] for an empty input.
 */
private fun median(values: List<Float>): Float {
    if (values.isEmpty()) {
        return MIN_LABEL_PROPAGATION_SIGMA
    }
    val sorted = values.sorted()
    val middleIndex = sorted.size / 2
    return if (sorted.size % 2 == 0) {
        (sorted[middleIndex - 1] + sorted[middleIndex]) * 0.5f
    } else {
        sorted[middleIndex]
    }
}

/**
 * One weighted neighbouring vertex in the propagation graph.
 *
 * @param index neighbouring vertex index.
 * @param weight propagation edge weight.
 */
private data class WeightedNeighbour(
    val index: Int,
    val weight: Float,
)

/**
 * One neighbour candidate with precomputed squared distance.
 *
 * @param index neighbour index.
 * @param distanceSquared squared distance from the source point.
 */
private data class IndexedDistance(
    val index: Int,
    val distanceSquared: Float,
)

private const val DEFAULT_LABEL_PROPAGATION_NEIGHBOUR_COUNT = 8
private const val DEFAULT_LABEL_PROPAGATION_MAX_ITERATIONS = 64
private const val MIN_LABEL_PROPAGATION_SIGMA = 0.05f
