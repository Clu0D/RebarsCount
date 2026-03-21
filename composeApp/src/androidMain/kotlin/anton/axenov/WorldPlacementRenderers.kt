package anton.axenov

import com.google.ar.core.Anchor
import com.google.ar.core.Pose
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.node.CubeNode
import kotlin.math.max
import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3

/**
 * Draws one managed zone polygon and all of its sampled points.
 *
 * @param sceneView active SceneView.
 * @param zone zone data to draw.
 * @return created anchor nodes for polygon edges and sampled points.
 */
fun drawZone(
    sceneView: ARSceneView,
    zone: Zone,
): List<AnchorNode> {
    val polygonNodes = createPolygonMarkerNodes(
        sceneView = sceneView,
        polygonPoints = zone.polygonPoints,
    )
    val pointNodes = createWorldPointMarkerNodes(
        sceneView = sceneView,
        worldPoints = zone.sampledPoints,
    )
    return polygonNodes + pointNodes
}

/**
 * Creates cube markers for world-space sample points and adds them to the scene.
 *
 * @param sceneView active SceneView.
 * @param worldPoints world-space points to visualize.
 * @return created anchor nodes, one per point.
 */
fun createWorldPointMarkerNodes(
    sceneView: ARSceneView,
    worldPoints: List<Vector3>,
): List<AnchorNode> {
    val session = sceneView.session ?: return emptyList()
    return worldPoints.map { worldPoint ->
        val anchor = session.createAnchor(Pose.makeTranslation(worldPoint.x, worldPoint.y, worldPoint.z))
        createWorldPointMarkerAnchorNode(sceneView, anchor)
    }
}

/**
 * Creates polygon edge markers from ordered world-space points and adds them to the scene.
 *
 * Consecutive points are connected and the last point is connected back to the first one.
 *
 * @param sceneView active SceneView.
 * @param polygonPoints polygon points in world coordinates.
 * @return created anchor nodes, one per rendered edge.
 */
fun createPolygonMarkerNodes(
    sceneView: ARSceneView,
    polygonPoints: List<Vector3>,
): List<AnchorNode> {
    if (polygonPoints.size < 3) {
        return emptyList()
    }
    val session = sceneView.session ?: return emptyList()
    val lengths = polygonPoints.indices.map { index ->
        val start = polygonPoints[index]
        val end = polygonPoints[(index + 1) % polygonPoints.size]
        (end - start).length
    }
    val averageLength = lengths.average().toFloat()
    val edgeThickness = max(averageLength * LINE_THICKNESS_FACTOR, LINE_MIN_THICKNESS_METERS)
        .coerceAtMost(LINE_MAX_THICKNESS_METERS)

    return polygonPoints.indices.mapNotNull { index ->
        val start = polygonPoints[index]
        val end = polygonPoints[(index + 1) % polygonPoints.size]
        val edgeVector = end - start
        val edgeLength = edgeVector.length
        if (edgeLength <= MIN_EDGE_LENGTH_METERS) {
            return@mapNotNull null
        }
        val edgeDirection = edgeVector / edgeLength
        val midpoint = (start + end) / 2f
        val edgeRotation = Quaternion.fromVectors(Vector3(1f, 0f, 0f), edgeDirection.normalized())
        val edgeAnchor = session.createAnchor(
            Pose(
                floatArrayOf(midpoint.x, midpoint.y, midpoint.z),
                floatArrayOf(edgeRotation.x, edgeRotation.y, edgeRotation.z, edgeRotation.w),
            ),
        )
        val anchorNode = AnchorNode(sceneView.engine, edgeAnchor)
        val edgeNode = CubeNode(
            engine = sceneView.engine,
            size = dev.romainguy.kotlin.math.Float3(edgeLength, edgeThickness, edgeThickness),
        )
        anchorNode.addChildNode(edgeNode)
        sceneView.addChildNode(anchorNode)
        anchorNode
    }
}

/**
 * Creates a marker anchor node for one world-space point and adds it to the scene.
 *
 * @param sceneView active SceneView.
 * @param anchor anchor positioned at point coordinates.
 * @return created anchor node.
 */
private fun createWorldPointMarkerAnchorNode(
    sceneView: ARSceneView,
    anchor: Anchor,
): AnchorNode {
    val anchorNode = AnchorNode(sceneView.engine, anchor)
    val pointCube = CubeNode(
        engine = sceneView.engine,
        size = dev.romainguy.kotlin.math.Float3(
            POINT_MARKER_SIZE_METERS,
            POINT_MARKER_SIZE_METERS,
            POINT_MARKER_SIZE_METERS,
        ),
    )
    anchorNode.addChildNode(pointCube)
    sceneView.addChildNode(anchorNode)
    return anchorNode
}

private const val POINT_MARKER_SIZE_METERS = 0.015f
private const val LINE_THICKNESS_FACTOR = 0.02f
private const val LINE_MIN_THICKNESS_METERS = 0.003f
private const val LINE_MAX_THICKNESS_METERS = 0.03f
private const val MIN_EDGE_LENGTH_METERS = 0.001f
