package anton.axenov

import korlibs.math.geom.Vector3F
import kotlin.collections.component1
import kotlin.collections.plusAssign
import kotlin.collections.set
import kotlin.math.sqrt

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
 * @param maxFreeWorldPoints maximal number of unresolved pairwise hypotheses retained between resolve calls.
 * @param newWorldPointSurvivalBonus temporary priority bonus assigned to recently added hypotheses.
 * @param componentDissolveDistanceMeters distance below which resolved components are considered duplicates.
 */
class WorldPointHypothesisResolver(
    minNormalizationConfidence: Float = 0.0f,
    private val maxNormalizationConfidence: Float = 0.4f,
    clusterRadiusMeters: Float = DEFAULT_CLUSTER_RADIUS_METERS,
    conflictConfidenceThreshold: Float = DEFAULT_CONFLICT_CONFIDENCE_THRESHOLD,
    private val replacementImprovementEpsilon: Float = DEFAULT_REPLACEMENT_IMPROVEMENT_EPSILON,
    private val minSupportEdgesForReplacement: Int = DEFAULT_MIN_SUPPORT_EDGES_FOR_REPLACEMENT,
    private val maxFreeWorldPoints: Int = DEFAULT_MAX_FREE_WORLD_POINTS,
    private val newWorldPointSurvivalBonus: Float = DEFAULT_NEW_WORLD_POINT_SURVIVAL_BONUS,
    private val componentDissolveDistanceMeters: Float = DEFAULT_COMPONENT_DISSOLVE_DISTANCE_METERS,
) {
    val worldPoints = mutableSetOf<WorldPoint>()
    val resolvedComponents = mutableListOf<HypothesisComponent>()
    var worldPointByObservations = mutableMapOf<Set<ZoneTriangulationPoint>, WorldPoint>()
    private val additionGenerationByWorldPoint = mutableMapOf<WorldPoint, Long>()
    private var currentGeneration = 0L
    private val baseMinNormalizationConfidence = minNormalizationConfidence
    private val baseClusterRadiusMeters = clusterRadiusMeters
    private val baseConflictConfidenceThreshold = conflictConfidenceThreshold
    private var currentThresholds = dynamicTriangulationThresholds(
        acceptedFrameCount = 0,
        minNormalizationConfidence = baseMinNormalizationConfidence,
        maxNormalizationConfidence = maxNormalizationConfidence,
        clusterRadiusMeters = baseClusterRadiusMeters,
        conflictConfidenceThreshold = baseConflictConfidenceThreshold,
        additionConfidenceThreshold = MIN_ADDITION_THRESHOLD,
        componentConfidenceThreshold = MIN_COMPONENT_THRESHOLD,
    )

    init {
        require(maxFreeWorldPoints >= 0) {
            "maxFreeWorldPoints must not be negative"
        }
        require(newWorldPointSurvivalBonus >= 0f) {
            "newWorldPointSurvivalBonus must not be negative"
        }
        require(componentDissolveDistanceMeters >= 0f) {
            "componentDissolveDistanceMeters must not be negative"
        }
    }

    /**
     * WorldPoints with normalized confidence.
     */
    private fun normalize(newWorldPoints: List<WorldPoint>): List<WorldPoint> {
        val confidenceRange = maxNormalizationConfidence - currentThresholds.minNormalizationConfidence
        return newWorldPoints.map { worldPoint ->
            worldPoint.copy(
                confidence = ((worldPoint.confidence - currentThresholds.minNormalizationConfidence) / confidenceRange)
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
     * @param acceptedFrameCount number of accepted frames including the current frame.
     * @param candidateComponentsByObservation components selected by ray proximity before edge generation.
     * @return resolved dense components ordered by descending confidence.
     */
    fun resolve(
        newWorldPoints: List<WorldPoint>,
        acceptedFrameCount: Int = (currentGeneration + 1L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
        candidateComponentsByObservation: Map<ZoneTriangulationPoint, List<HypothesisComponent>> = emptyMap(),
    ): List<WorldPoint> {
        require(acceptedFrameCount >= 0) {
            "acceptedFrameCount must not be negative"
        }
        currentGeneration++
        currentThresholds = dynamicTriangulationThresholds(
            acceptedFrameCount = acceptedFrameCount,
            minNormalizationConfidence = baseMinNormalizationConfidence,
            maxNormalizationConfidence = maxNormalizationConfidence,
            clusterRadiusMeters = baseClusterRadiusMeters,
            conflictConfidenceThreshold = baseConflictConfidenceThreshold,
            additionConfidenceThreshold = MIN_ADDITION_THRESHOLD,
            componentConfidenceThreshold = MIN_COMPONENT_THRESHOLD,
        )
        val normalizedPoints = normalize(newWorldPoints)
        worldPoints += normalizedPoints
        normalizedPoints.forEach { worldPoint ->
            additionGenerationByWorldPoint[worldPoint] = currentGeneration
        }
        worldPointByObservations = buildWorldPointByObservations()
        val releasedComponentPoints = auditAndDissolveComponents(normalizedPoints)
        val directlyMergedObservations = mergeObservationsIntoRayCandidateComponents(
            candidateComponentsByObservation,
        )

        (normalizedPoints + releasedComponentPoints)
            .distinct()
            .filterNot { point ->
                point.parentPoints.any { observation -> observation in directlyMergedObservations }
            }
            .sortedByDescending { it.confidence }
            .forEach { newPoint ->
                mergeNewPointIntoResolvedComponents(newPoint)?.let { component ->
                    removeUsedObservations(component)
                }
            }
        trimFreeWorldPoints()

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
     * Tries new observations against components preselected by viewing-ray proximity.
     *
     * @param candidateComponentsByObservation components ordered by distance to each observation ray.
     * @return observations successfully included in a component.
     */
    private fun mergeObservationsIntoRayCandidateComponents(
        candidateComponentsByObservation: Map<ZoneTriangulationPoint, List<HypothesisComponent>>,
    ): Set<ZoneTriangulationPoint> {
        val mergedObservations = mutableSetOf<ZoneTriangulationPoint>()
        candidateComponentsByObservation.forEach { (observation, candidateComponents) ->
            candidateComponents
                .filter { component -> component in resolvedComponents }
                .firstOrNull { component ->
                    component.tryAddingOrReplacingCandidate(observation).also { merged ->
                        if (merged) {
                            component.recomputeCenterAndConfidence()
                            removeUsedObservations(component)
                        }
                    }
                }
                ?.let { mergedObservations += observation }
        }
        return mergedObservations
    }

    /**
     * Audits resolved components and returns dissolved edges to the ordinary resolution pipeline.
     *
     * Weak components are dissolved. Both components of every impossibly close pair are dissolved
     * so their observations can be glued into one component. When component count exceeds the
     * maximum observation count on participating frames, weakest excess components are dissolved.
     *
     * @param newWorldPoints newly added normalized hypotheses included in the frame-count audit.
     * @return released component edges that should be processed like newly added hypotheses.
     */
    internal fun auditAndDissolveComponents(newWorldPoints: List<WorldPoint> = emptyList()): List<WorldPoint> {
        if (resolvedComponents.isEmpty()) {
            return emptyList()
        }
        val componentsToDissolve = resolvedComponents
            .filterTo(mutableSetOf()) { component -> component.isBad() }

        resolvedComponents.forEachIndexed { firstIndex, firstComponent ->
            resolvedComponents
                .drop(firstIndex + 1)
                .filter { secondComponent ->
                    (firstComponent.center - secondComponent.center).length <= componentDissolveDistanceMeters
                }
                .forEach { secondComponent ->
                    componentsToDissolve += firstComponent
                    componentsToDissolve += secondComponent
                }
        }

        val maximumFrameObservationCount = (
            resolvedComponents.flatMap { component -> component.curObservations() } +
                newWorldPoints.flatMap { point -> point.parentPoints }
            )
            .groupingBy { observation -> observation.segmentation }
            .eachCount()
            .values
            .maxOrNull()
            ?: 0
        val remainingComponents = resolvedComponents.filterNot { component -> component in componentsToDissolve }
        val excessComponentCount = (remainingComponents.size - maximumFrameObservationCount).coerceAtLeast(0)
        componentsToDissolve += remainingComponents
            .sortedBy { component -> component.confidence }
            .take(excessComponentCount)

        return dissolveComponents(componentsToDissolve)
    }

    /**
     * Removes components and returns their support edges to the unresolved pool.
     *
     * @param components components selected by the audit.
     * @return released support edges.
     */
    private fun dissolveComponents(components: Set<HypothesisComponent>): List<WorldPoint> {
        if (components.isEmpty()) {
            return emptyList()
        }
        val releasedPoints = components
            .flatMap { component -> component.curPoints() }
            .distinct()
        resolvedComponents.removeAll(components)
        worldPoints += releasedPoints
        releasedPoints.forEach { point ->
            additionGenerationByWorldPoint[point] = currentGeneration
        }
        worldPointByObservations = buildWorldPointByObservations()
        return releasedPoints
    }

    /**
     * Removes points associated with observations from one component.
     */
    fun removeUsedObservations(component: HypothesisComponent) {
        val usedObservations = component.curObservations()
        val removedPoints = worldPoints.filterTo(mutableSetOf()) { point ->
            point in component.curPoints() ||
                    point.parentPoints.any { it in usedObservations }
        }
        worldPoints.removeAll(removedPoints)
        additionGenerationByWorldPoint.keys.removeAll(removedPoints)
        worldPointByObservations = buildWorldPointByObservations()
    }

    /**
     * Evicts lowest-priority unresolved hypotheses when the free pool exceeds its configured limit.
     *
     * Priority combines normalized confidence with a survival bonus that decays as
     * `newWorldPointSurvivalBonus / (age + 1)`. This lets recent hypotheses displace weak stale
     * hypotheses while preserving sufficiently strong older hypotheses.
     */
    internal fun trimFreeWorldPoints() {
        if (worldPoints.size <= maxFreeWorldPoints) {
            return
        }
        val retainedPoints = worldPoints
            .sortedWith(
                compareByDescending<WorldPoint> { point -> freeWorldPointRetentionPriority(point) }
                    .thenByDescending { point -> additionGenerationByWorldPoint[point] ?: 0L }
                    .thenByDescending { point -> point.confidence },
            )
            .take(maxFreeWorldPoints)
            .toSet()
        worldPoints.retainAll(retainedPoints)
        additionGenerationByWorldPoint.keys.retainAll(retainedPoints)
        worldPointByObservations = buildWorldPointByObservations()
    }

    /**
     * Calculates current eviction priority for one unresolved hypothesis.
     *
     * @param worldPoint unresolved pairwise hypothesis.
     * @return confidence plus age-decaying survival bonus.
     */
    internal fun freeWorldPointRetentionPriority(worldPoint: WorldPoint): Float {
        val additionGeneration = additionGenerationByWorldPoint[worldPoint] ?: 0L
        val age = (currentGeneration - additionGeneration).coerceAtLeast(0L)
        return worldPoint.confidence + newWorldPointSurvivalBonus / (age + 1L)
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
            .filter { component -> (newPoint.position - component.center).length <= currentThresholds.clusterRadiusMeters }
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

            val supportEdgeCount = supportByPair.size
            val averageSupportWeight = totalWeight / supportEdgeCount
            confidence = averageSupportWeight * sqrt(supportEdgeCount.toFloat())
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
                if (pairSupport == null) {
                    if (candidateObservation.hasForbiddenEdgeTo(selectedObservation)) {
                        conflictingObservations += selectedObservation
                    }
                    return@forEach
                }
                if (
                    pairSupport.confidence >= currentThresholds.conflictConfidenceThreshold &&
                    isPointClose(pairSupport)
                ) {
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
                    if (supportSum / supportEdges.size < currentThresholds.additionConfidenceThreshold)
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
            return (worldPoint.position - center).length <= currentThresholds.clusterRadiusMeters
        }

        /**
         * Checks if component is too bad to approve it
         */
        fun isBad(): Boolean {
            return confidence < currentThresholds.componentConfidenceThreshold ||
                curPoints().size < MIN_COMPONENT_SIZE
        }
    }
}

private fun sumConfidenceByEdges(points: List<WorldPoint>): Float {
    return points.map { it.confidence }.sum()
}

private const val DEFAULT_CLUSTER_RADIUS_METERS = 0.03f
private const val DEFAULT_CONFLICT_CONFIDENCE_THRESHOLD = 0.7f
private const val MIN_COMPONENT_THRESHOLD = 1.5f
private const val MIN_ADDITION_THRESHOLD = 0.5f
private const val MIN_COMPONENT_SIZE = 4
private const val DEFAULT_REPLACEMENT_IMPROVEMENT_EPSILON = 1e-4f
private const val DEFAULT_MIN_SUPPORT_EDGES_FOR_REPLACEMENT = 2
private const val DEFAULT_MAX_FREE_WORLD_POINTS = 5_000
private const val DEFAULT_NEW_WORLD_POINT_SURVIVAL_BONUS = 0.15f
private const val DEFAULT_COMPONENT_DISSOLVE_DISTANCE_METERS = 0.005f
