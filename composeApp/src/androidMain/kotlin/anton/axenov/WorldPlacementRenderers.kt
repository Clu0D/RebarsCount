package anton.axenov

import com.google.ar.core.Anchor
import com.google.ar.core.Pose
import com.google.ar.core.Session
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.math.Color
import io.github.sceneview.node.CubeNode
import kotlin.math.max
import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3

/**
 * Draws static geometry nodes of one zone: polygon edges, sampled points and 3D bounding box.
 *
 * @param sceneView active SceneView.
 * @param zone zone data to draw.
 * @return created static anchor nodes.
 */
fun drawZoneStaticNodes(
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
    val zoneBoundingBoxNodes = createZoneBoundingBoxNodes(
        sceneView = sceneView,
        zone = zone,
    )
    return polygonNodes + pointNodes + zoneBoundingBoxNodes
}

/**
 * Draws one short world-space line that starts at snapshot camera position and points
 * in the snapshot camera forward direction.
 *
 * @param sceneView active SceneView.
 * @param frameSnapshot captured frame snapshot with camera pose.
 * @return created anchor node or null when AR session is unavailable.
 */
fun drawSnapshotCameraDirectionNode(
    sceneView: ARSceneView,
    frameSnapshot: DetectionFrameSnapshot,
): AnchorNode? {
    val session = sceneView.session ?: return null
    val cameraPose = frameSnapshot.cameraPose
    val lineStart = Vector3(
        cameraPose.tx(),
        cameraPose.ty(),
        cameraPose.tz(),
    )
    val lineEndWorld = cameraPose.transformPoint(
        floatArrayOf(0f, 0f, -SNAPSHOT_DIRECTION_LINE_LENGTH_METERS),
    )
    val lineEnd = Vector3(
        lineEndWorld[0],
        lineEndWorld[1],
        lineEndWorld[2],
    )
    val lineMaterial = sceneView.materialLoader.createColorInstance(
        Color(1f, 0.55f, 0f, 1f),
    )
    return createEdgeAnchorNode(
        sceneView = sceneView,
        session = session,
        start = lineStart,
        end = lineEnd,
    ) { edgeLength ->
        CubeNode(
            engine = sceneView.engine,
            size = dev.romainguy.kotlin.math.Float3(
                edgeLength,
                SNAPSHOT_DIRECTION_LINE_THICKNESS_METERS,
                SNAPSHOT_DIRECTION_LINE_THICKNESS_METERS,
            ),
            materialInstance = lineMaterial,
        )
    }
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
    val materialInstance = sceneView.materialLoader.createColorInstance(
        Color(1f, 1f, 1f, 1f),
    )
    return worldPoints.map { worldPoint ->
        val anchor = session.createAnchor(Pose.makeTranslation(worldPoint.x, worldPoint.y, worldPoint.z))
        createWorldPointMarkerAnchorNode(sceneView, anchor, POINT_MARKER_SIZE_METERS, materialInstance)
    }
}

/**
 * Creates marker nodes for reconstructed world points and adds them to the scene.
 *
 * @param sceneView active SceneView.
 * @param worldPoints world-space points to visualize.
 * @return created anchor nodes, one per point.
 */
fun createServerWorldPointMarkerNodes(
    sceneView: ARSceneView,
    worldPoints: List<Vector3>,
): List<AnchorNode> {
    val session = sceneView.session ?: return emptyList()
    val materialInstance = sceneView.materialLoader.createColorInstance(
        Color(0.1f, 0.9f, 0.2f, 1f),
    )
    return worldPoints.map { worldPoint ->
        val anchor = session.createAnchor(Pose.makeTranslation(worldPoint.x, worldPoint.y, worldPoint.z))
        createWorldPointMarkerAnchorNode(sceneView, anchor, SERVER_POINT_MARKER_SIZE_METERS, materialInstance)
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
        createEdgeAnchorNode(
            sceneView = sceneView,
            session = session,
            start = start,
            end = end,
        ) { edgeLength ->
            CubeNode(
                engine = sceneView.engine,
                size = dev.romainguy.kotlin.math.Float3(edgeLength, edgeThickness, edgeThickness),
            )
        }
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
    markerSizeMeters: Float,
    materialInstance: com.google.android.filament.MaterialInstance,
): AnchorNode {
    val anchorNode = AnchorNode(sceneView.engine, anchor)
    val pointCube = CubeNode(
        engine = sceneView.engine,
        size = dev.romainguy.kotlin.math.Float3(
            markerSizeMeters,
            markerSizeMeters,
            markerSizeMeters,
        ),
        materialInstance = materialInstance,
    )
    anchorNode.addChildNode(pointCube)
    sceneView.addChildNode(anchorNode)
    return anchorNode
}


/**
 * Draws zone 3D bounding box as thin blue wireframe edges.
 *
 * @param sceneView active SceneView.
 * @param zone zone to visualize.
 * @return created anchor nodes, one per rendered edge.
 */
private fun createZoneBoundingBoxNodes(
    sceneView: ARSceneView,
    zone: Zone,
): List<AnchorNode> {
    val session = sceneView.session ?: return emptyList()
    val box = zone.boundingBox
    val points = listOf(
        Vector3(box.minX, box.minY, box.minZ),
        Vector3(box.maxX, box.minY, box.minZ),
        Vector3(box.maxX, box.maxY, box.minZ),
        Vector3(box.minX, box.maxY, box.minZ),
        Vector3(box.minX, box.minY, box.maxZ),
        Vector3(box.maxX, box.minY, box.maxZ),
        Vector3(box.maxX, box.maxY, box.maxZ),
        Vector3(box.minX, box.maxY, box.maxZ),
    )
    val edgeIndices = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 0,
        4 to 5, 5 to 6, 6 to 7, 7 to 4,
        0 to 4, 1 to 5, 2 to 6, 3 to 7,
    )
    val blueMaterial = sceneView.materialLoader.createColorInstance(
        Color(0f, 0f, 1f, 1f),
    )
    return edgeIndices.mapNotNull { (startIndex, endIndex) ->
        val start = points[startIndex]
        val end = points[endIndex]
        createEdgeAnchorNode(
            sceneView = sceneView,
            session = session,
            start = start,
            end = end,
        ) { edgeLength ->
            CubeNode(
                engine = sceneView.engine,
                size = dev.romainguy.kotlin.math.Float3(
                    edgeLength,
                    BOUNDING_BOX_LINE_THICKNESS_METERS,
                    BOUNDING_BOX_LINE_THICKNESS_METERS,
                ),
                materialInstance = blueMaterial,
            )
        }
    }
}

/**
 * Creates and attaches one oriented edge node between two world-space points.
 *
 * @param sceneView active SceneView.
 * @param session ARCore session used to create an anchor.
 * @param start edge start in world coordinates.
 * @param end edge end in world coordinates.
 * @param edgeNodeFactory builder that creates a node by edge length.
 * @return created anchor node or null when edge is too short.
 */
private fun createEdgeAnchorNode(
    sceneView: ARSceneView,
    session: Session,
    start: Vector3,
    end: Vector3,
    edgeNodeFactory: (edgeLength: Float) -> CubeNode,
): AnchorNode? {
    val edgeVector = end - start
    val edgeLength = edgeVector.length
    if (edgeLength <= MIN_EDGE_LENGTH_METERS) {
        return null
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
    anchorNode.addChildNode(edgeNodeFactory(edgeLength))
    sceneView.addChildNode(anchorNode)
    return anchorNode
}

private const val POINT_MARKER_SIZE_METERS = 0.005f
private const val SERVER_POINT_MARKER_SIZE_METERS = 0.012f
private const val LINE_THICKNESS_FACTOR = 0.02f
private const val LINE_MIN_THICKNESS_METERS = 0.003f
private const val LINE_MAX_THICKNESS_METERS = 0.03f
private const val MIN_EDGE_LENGTH_METERS = 0.001f
private const val BOUNDING_BOX_LINE_THICKNESS_METERS = 0.0015f
private const val SNAPSHOT_DIRECTION_LINE_LENGTH_METERS = 0.1f
private const val SNAPSHOT_DIRECTION_LINE_THICKNESS_METERS = 0.003f
