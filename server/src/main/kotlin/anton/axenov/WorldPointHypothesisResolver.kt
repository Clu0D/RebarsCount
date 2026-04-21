package anton.axenov

import korlibs.math.geom.Vector3F
import kotlin.collections.component1
import kotlin.collections.plusAssign
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
 * When new points are added they are tried to be added to existing hypothesis,
 * then the main resolve is used for all unused points.
 *
 * @param clusterRadiusMeters maximal allowed distance from component center for edge to be added.
 * @param conflictConfidenceThreshold normalized confidence below which edge is treated as a conflict.
 * @param replacementImprovementEpsilon minimal improvement required to perform a replacement move.
 * @param minSupportEdgesForReplacement minimal number of strong edges required before replacement.
 */
class WorldPointHypothesisResolver(
    private val minNormalizationConfidence: Float = 0.0f,
    private val maxNormalizationConfidence: Float = 0.8f,
    private val clusterRadiusMeters: Float = DEFAULT_CLUSTER_RADIUS_METERS,
    private val conflictConfidenceThreshold: Float = DEFAULT_CONFLICT_CONFIDENCE_THRESHOLD,
    private val replacementImprovementEpsilon: Float = DEFAULT_REPLACEMENT_IMPROVEMENT_EPSILON,
    private val minSupportEdgesForReplacement: Int = DEFAULT_MIN_SUPPORT_EDGES_FOR_REPLACEMENT,
) {
    val worldPoints = mutableSetOf<WorldPoint>()
    val resolvedComponents = mutableListOf<HypothesisComponent>()
    var worldPointByObservations = mutableMapOf<Set<ZoneTriangulationPoint>, WorldPoint>()

    /**
     * WorldPoints with normalized confidence.
     */
    private fun normalize(newWorldPoints: List<WorldPoint>): List<WorldPoint> {
        val confidenceRange = maxNormalizationConfidence - minNormalizationConfidence
        return newWorldPoints.map { worldPoint ->
            worldPoint.copy(
                confidence = ((worldPoint.confidence - minNormalizationConfidence) / confidenceRange)
                    .coerceIn(0f, 1f)
            )
        }
    }

    /**
     * Builds lookup map for points by observation pairs.
     */
    private fun buildWorldPointByObservations(): MutableMap<Set<ZoneTriangulationPoint>, WorldPoint> {
        val map = mutableMapOf<Set<ZoneTriangulationPoint>, WorldPoint>()
        worldPoints.forEach { point ->
            map += point.parentPoints to point
        }
        return map
    }

    /**
     * Resolves 2-view [WorldPoint]s into multi-view components.
     *
     * @param newWorldPoints raw triangulated points.
     * @return resolved dense components ordered by descending confidence.
     */
    @Synchronized
    fun resolve(newWorldPoints: List<WorldPoint>): List<WorldPoint> {
        val normalizedPoints = normalize(newWorldPoints)
        worldPoints += normalizedPoints
        worldPointByObservations = buildWorldPointByObservations()

        normalizedPoints.forEach { newPoint ->
            mergeNewPointIntoResolvedComponents(newPoint)?.let { component ->
                removeUsedObservations(component)
            }
        }

        val usedPoints = mutableSetOf<WorldPoint>()
        while (true) {
            val bestPoint = (worldPoints.toSet() - usedPoints).maxByOrNull { it.confidence } ?: break
            val component = growComponent(bestPoint, worldPoints)
            if (component.isBad()) {
                usedPoints += bestPoint
                continue
            }

            resolvedComponents += component
            removeUsedObservations(component)
        }
        return resolvedComponents.map { it.toWorldPoint() }.sortedByDescending { component -> component.confidence }
    }

    /**
     * Removes points associated with observations from one component.
     */
    fun removeUsedObservations(component: HypothesisComponent) {
        val usedObservations = component.curObservations()
        worldPoints.removeAll { point ->
            point in component.curPoints() ||
                    point.parentPoints.any { it in usedObservations }
        }
        worldPointByObservations = buildWorldPointByObservations()
    }

    /**
     * Tries to merge every new point into each previously resolved component.
     *
     * @param components mutable old components.
     * @param newPoint point to be absorbed.
     */
    private fun mergeNewPointIntoResolvedComponents(
        newPoint: WorldPoint,
    ): HypothesisComponent? {
        val candidateComponents = resolvedComponents
            .filter { component -> (newPoint.position - component.center).length <= clusterRadiusMeters }
            .sortedBy { component -> (newPoint.position - component.center).length }

        candidateComponents.forEach { component ->
            val candidateObservations = newPoint.parentPoints
                .filterNot { observation -> observation in component.curObservations() }
                .filterNot { observation -> observation.segmentation in component.selectedByFrame }

            var added = false
            candidateObservations.forEach { candidateObservation ->
                if (component.tryAddingOrReplacingCandidate(candidateObservation)) {
                    component.recomputeCenterAndConfidence()
                    added = true
                }
            }
            if (added)
                return component
        }
        return null
    }

    /**
     * Greedily grows one component from [worldPoint].
     *
     * @param worldPoint strongest remaining hypothesis.
     * @param candidatePoints points allowed for growth.
     * @return completed component after add/replace hill climbing converges.
     */
    private fun growComponent(worldPoint: WorldPoint, candidatePoints: Set<WorldPoint>): HypothesisComponent {
        val component = HypothesisComponent(worldPoint = worldPoint)

        loop@ while (true) {
            val closestObservations = component.getClosestObservations(candidatePoints)
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
    inner class HypothesisComponent(worldPoint: WorldPoint) {
        val selectedByFrame = worldPoint.parentPoints.associateBy { p -> p.segmentation }.toMutableMap()
        val supportByPair = mutableMapOf(worldPoint.parentPoints to worldPoint)
        var center = worldPoint.position
        var confidence = worldPoint.confidence

        fun curPoints(): Set<WorldPoint> = supportByPair.values.toSet()
        fun curObservations(): Set<ZoneTriangulationPoint> = selectedByFrame.values.toSet()

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
         * Converts to one client-facing resolved world point.
         */
        fun toWorldPoint(): WorldPoint {
            return WorldPoint(
                position = center,
                parentPoints = curObservations(),
                confidence = confidence,
            )
        }

        /**
         * Find closest WorldPoints to the component center, that are
         *  not in component
         *  have at one frame that is already used in component
         *  sorted by distance
         * and get the observation that is still not in component
         */
        fun getClosestObservations(candidatePoints: Set<WorldPoint>): Set<ZoneTriangulationPoint> {
            val selectedObservations = curObservations()
            val candidateDistanceByObservation = mutableMapOf<ZoneTriangulationPoint, Float>()

            candidatePoints
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
            supportEdges: List<WorldPoint>,
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

        /**
         * Tries to add or replace [candidateObservation].
         *
         * @param candidateObservation candidate observation.
         * @return true when the component changed.
         */
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
                    val replacedSupport = sumConfidenceByEdges(
                        curPoints().filter { point -> replacedObservation in point.parentPoints }
                    )
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
        private fun isPointClose(worldPoint: WorldPoint): Boolean {
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
