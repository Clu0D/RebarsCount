package anton.axenov

import korlibs.math.geom.Vector3F
import kotlin.collections.component1
import kotlin.collections.set
import kotlin.math.max

/**
 * Greedily combines 2-view [WorldPoint] hypotheses to multi-view point components.
 *
 * Each raw [WorldPoint] is an edge between 2 observations from 2 different frames.
 * The resolver grows one component at a time by:
 * 1. choosing the strongest remaining unused edge as base,
 * 2. iteratively finding the best compatible observations from new frames,
 * 3. either adding new or replacing old observation, trying to make the most consistent set of observations.
 *
 * @param clusterRadiusMeters maximal allowed distance from component center for edge to be added.
 * @param conflictConfidenceThreshold normalized confidence below which edge is treated as a conflict.
 * @param replacementImprovementEpsilon minimal improvement required to perform a replacement move.
 * @param minSupportEdgesForReplacement minimal number of strong edges required before replacement.
 */
class WorldPointHypothesisResolver(
    worldPoints: List<WorldPoint>,
    private val clusterRadiusMeters: Float = DEFAULT_CLUSTER_RADIUS_METERS,
    private val conflictConfidenceThreshold: Float = DEFAULT_CONFLICT_CONFIDENCE_THRESHOLD,
    private val replacementImprovementEpsilon: Float = DEFAULT_REPLACEMENT_IMPROVEMENT_EPSILON,
    private val minSupportEdgesForReplacement: Int = DEFAULT_MIN_SUPPORT_EDGES_FOR_REPLACEMENT,
) {
    /**
     * WorldPoints with normalized confidence.
     */
    val worldPoints: MutableSet<WorldPoint> = run {
        val validWorldPoints = worldPoints.filter { worldPoint -> worldPoint.parentPoints.size == 2 }
        if (validWorldPoints.isEmpty()) {
            emptyList()
        } else {
            val minConfidence = validWorldPoints.minOf { worldPoint -> worldPoint.confidence }
            val maxConfidence = validWorldPoints.maxOf { worldPoint -> worldPoint.confidence }
            val confidenceRange = maxConfidence - minConfidence
            if (confidenceRange <= 0e-6) {
                validWorldPoints
            } else {
                validWorldPoints.map { worldPoint ->
                    worldPoint.copy(
                        confidence = (worldPoint.confidence - minConfidence) / confidenceRange
                    )
                }
            }
        }
    }.toMutableSet()

    var worldPointByObservations = buildWorldPointByObservations()

    private fun buildWorldPointByObservations(): Map<Set<ZoneTriangulationPoint>, WorldPoint> {
        val map = mutableMapOf<Set<ZoneTriangulationPoint>, WorldPoint>()
        worldPoints.forEach { point ->
            map += point.parentPoints to point
        }
        return map.toMap()
    }

    /**
     * Component that indicates one real-world point.
     *
     * @param position weighted component center.
     * @param confidence aggregated component confidence in range `[0, 1]`.
     * @param selectedObservations chosen per-frame observations supporting this point.
     * @param supportWorldPoints pairwise support edges selected into the component.
     */
    data class ResolvedWorldPointComponent(
        val position: Vector3F,
        val confidence: Float,
        val selectedObservations: Set<ZoneTriangulationPoint>,
        val supportWorldPoints: Set<WorldPoint>,
    )

    /**
     * Resolves 2-view [WorldPoint]s into multi-view components.
     *
     * @param worldPoints raw triangulated points.
     * @return resolved dense components ordered by descending confidence.
     */
    fun resolve(): List<ResolvedWorldPointComponent> {
        val resolvedComponents = mutableListOf<ResolvedWorldPointComponent>()

        while (true) {
            val bestPoint = worldPoints.maxByOrNull { hypothesis -> hypothesis.confidence } ?: break
            val component = growComponent(bestPoint)
            if (component.isBad()) {
                worldPoints.remove(bestPoint)
                continue
            }

            resolvedComponents += component.toResolvedComponent()

            val usedObservations = component.selectedByFrame.values.toSet()
            worldPoints.removeAll { point ->
                point in component.supportByPair.values ||
                        point.parentPoints.any { it in usedObservations }
            }
            worldPointByObservations = buildWorldPointByObservations()
        }

        return resolvedComponents.sortedByDescending { component -> component.confidence }
    }

    /**
     * Greedily grows one component from [worldPoint].
     *
     * @param worldPoint strongest remaining pairwise hypothesis.
     * @return completed component after add/replace hill climbing converges.
     */
    private fun growComponent(
        worldPoint: WorldPoint,
    ): HypothesisComponent {
        val component = HypothesisComponent(worldPoint)

        loop@ while (true) {
            val closestObservations = component.getClosestObservations()
            closestObservations.forEach { observation ->
                if (component.tryAddingOrReplacingCandidate(observation)) {
                    component.recomputeCenterAndConfidence()
                    continue@loop
                }
            }
            break@loop
        }
        return component
    }

    /**
     * Mutable component being greedily expanded.
     *
     * @param selectedByFrame selected observation for each frame currently participating in the component.
     * @param supportByPair chosen pairwise support edges between selected observations.
     * @param center current weighted component center.
     * @param confidence aggregated component confidence.
     */
    private inner class HypothesisComponent(worldPoint: WorldPoint) {
        val selectedByFrame = worldPoint.parentPoints.associateBy { p -> p.segmentation }.toMutableMap()
        val supportByPair = mutableMapOf(worldPoint.parentPoints to worldPoint)
        var center = worldPoint.position
        var confidence = worldPoint.confidence

        fun curPoints(): Set<WorldPoint> = supportByPair.values.toSet()
        fun curObservations() : Set<ZoneTriangulationPoint> = selectedByFrame.values.toSet()

        /**
         * Returns the strongest current support of [observation] to the rest of the component.
         *
         * @param observation selected observation already inside the component.
         * @return sum of normalized pairwise support confidences.
         */
        fun currentSupportOf(observation: ZoneTriangulationPoint): Float {
            return sumConfidenceByEdges(
                curPoints()
                    .filter { hypothesis ->
                        observation in hypothesis.parentPoints
                    })
        }

        /**
         * Recomputes weighted center and aggregated confidence from current support edges.
         */
        fun recomputeCenterAndConfidence() {
            if (supportByPair.isEmpty())
                return

            val totalWeight = curPoints().map { it.confidence }.sum()
            var weightedPositionSum = Vector3F(0f, 0f, 0f)
            curPoints().forEach { hypothesis ->
                weightedPositionSum += hypothesis.position * hypothesis.confidence
            }
            center = weightedPositionSum / totalWeight

            confidence = (sumConfidenceByEdges(curPoints().toList()) / max(1, supportByPair.size))
                .coerceIn(0f, 1f)
        }

        /**
         * Converts to public immutable output.
         */
        fun toResolvedComponent(): ResolvedWorldPointComponent {
            return ResolvedWorldPointComponent(
                position = center,
                confidence = confidence,
                selectedObservations = curObservations(),
                supportWorldPoints = curPoints(),
            )
        }

        /**
         * Find closest WorldPoints to the component center, that are
         *  not in component
         *  have at one frame that is already used in component
         *  sorted by distance
         * and get the observation that is still not in component
         */
        fun getClosestObservations(): Set<ZoneTriangulationPoint> {
            val selectedObservations = curObservations()
            val candidateDistanceByObservation = mutableMapOf<ZoneTriangulationPoint, Float>()

            // todo add some spatial map to find closest set faster
            worldPoints
                .filterNot { worldPoint -> worldPoint in curPoints() }
                .filter { worldPoint -> worldPoint.parentPoints.any { observation -> observation in selectedObservations } }
                .forEach { worldPoint ->
                    val distanceToCenter = (worldPoint.position - center).length
                    worldPoint.parentPoints
                        .filterNot { observation -> observation in selectedObservations }
                        .filterNot { observation -> observation.segmentation in selectedByFrame }
                        .forEach { observation ->
                            val previousDistance = candidateDistanceByObservation[observation]
                            if (previousDistance == null || distanceToCenter < previousDistance) {
                                candidateDistanceByObservation[observation] = distanceToCenter
                            }
                        }
                }

            return candidateDistanceByObservation.entries
                .sortedBy { (_, distance) -> distance }
                .map { (observation, _) -> observation }
                .toCollection(linkedSetOf())
        }

        fun addObservation(observation: ZoneTriangulationPoint, supportEdges: List<WorldPoint>) {
            selectedByFrame[observation.segmentation] = observation
            supportEdges.forEach { supportEdge ->
                supportByPair[supportEdge.parentPoints] = supportEdge
            }
        }

        fun replaceObservation(
            observation: ZoneTriangulationPoint,
            replacedObservation: ZoneTriangulationPoint,
            supportEdges: List<WorldPoint>
        ) {
            selectedByFrame.remove(replacedObservation.segmentation)
            supportByPair.entries.removeAll { (_, hypothesis) ->
                replacedObservation in hypothesis.parentPoints
            }
            selectedByFrame[observation.segmentation] = observation
            supportEdges.forEach { supportEdge ->
                supportByPair[supportEdge.parentPoints] = supportEdge
            }
        }

        fun tryAddingOrReplacingCandidate(candidateObservation: ZoneTriangulationPoint): Boolean {
            if (candidateObservation.segmentation in selectedByFrame)
                return false

            val supportEdges = mutableListOf<WorldPoint>()
            val conflictingObservations = mutableListOf<ZoneTriangulationPoint>()

            curObservations().forEach { selectedObservation ->
                val pairSupport = worldPointByObservations[setOf(candidateObservation, selectedObservation)]
                    ?: return@forEach
                if (pairSupport.confidence >= conflictConfidenceThreshold && isPointClose(pairSupport)) {
                    supportEdges += pairSupport
                } else {
                    conflictingObservations += selectedObservation
                }
            }

            if (supportEdges.isEmpty())
                return false

            val supportSum = sumConfidenceByEdges(supportEdges)
            return when (conflictingObservations.size) {
                0 -> {
                    if (supportSum / supportEdges.size < MIN_ADDITION_THRESHOLD)
                        return false

                    addObservation(candidateObservation, supportEdges)
                    true
                }

                1 -> {
                    if (supportEdges.size < minSupportEdgesForReplacement)
                        return false

                    val replacedObservation = conflictingObservations.single()
                    val replacedSupport = currentSupportOf(replacedObservation)
                    val improvement = supportSum - replacedSupport
                    if (improvement <= replacementImprovementEpsilon)
                        return false

                    replaceObservation(candidateObservation, replacedObservation, supportEdges)
                    true
                }

                else -> false
            }
        }

        /**
         * Is true when [worldPoint] lies close enough to [center] to belong to the component.
         */
        private fun isPointClose(
            worldPoint: WorldPoint
        ): Boolean {
            return (worldPoint.position - center).length <= clusterRadiusMeters
        }

        /**
         * Checks if component is too bad to approve it
         */
        fun isBad(): Boolean {
            return confidence < MIN_COMPONENT_THRESHOLD || curPoints().size < MIN_COMPONENT_SIZE
        }
    }
}

private fun sumConfidenceByEdges(points: List<WorldPoint>): Float {
    return points.map { it.confidence }.sum()
}

private const val DEFAULT_CLUSTER_RADIUS_METERS = 0.15f
private const val DEFAULT_CONFLICT_CONFIDENCE_THRESHOLD = 0.7f
private const val MIN_COMPONENT_THRESHOLD = 0.5f
private const val MIN_ADDITION_THRESHOLD = 0.3f
private const val MIN_COMPONENT_SIZE = 3
private const val DEFAULT_REPLACEMENT_IMPROVEMENT_EPSILON = 1e-4f
private const val DEFAULT_MIN_SUPPORT_EDGES_FOR_REPLACEMENT = 2
