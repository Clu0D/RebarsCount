package anton.axenov

import com.google.android.filament.MaterialInstance
import com.google.ar.core.Anchor
import com.google.ar.core.exceptions.FatalException
import com.google.ar.core.Pose
import com.google.ar.core.Session
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.node.CubeNode
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
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
            materialInstance = getUnlitMaterial(sceneView, 1f, 1f, 1f, 0.5f),
        )
    }
}


/**
 * Creates markers for world points and adds them to the scene.
 *
 * @param sceneView active SceneView.
 * @param worldPoints world-space points to visualize.
 * @param color point marker color.
 * @return created anchor nodes, one per point.
 */
fun createWorldPointMarkerNodes(
    sceneView: ARSceneView,
    worldPoints: List<Vector3>,
): List<AnchorNode> {
    val session = sceneView.session ?: return emptyList()
    return worldPoints.mapNotNull { worldPoint ->
        val anchor = createTranslationAnchor(
            session = session,
            worldPoint = worldPoint,
        ) ?: return@mapNotNull null
        createWorldPointMarkerAnchorNode(
            sceneView = sceneView,
            anchor = anchor,
            markerSizeMeters = POINT_MARKER_SIZE_METERS,
            materialInstance = getUnlitMaterial(sceneView, 1f, 1f, 1f, 0.5f),
        )
    }
}

/**
 * Creates markers for world points with confidence and adds them to the scene.
 *
 * @param sceneView active SceneView.
 * @param worldPoints world-space points to visualize.
 * @return created anchor nodes, one per point.
 */
fun createServerWorldPointMarkerNodes(
    sceneView: ARSceneView,
    worldPoints: List<ServerWorldPointDto>,
): List<AnchorNode> {
    val session = sceneView.session ?: return emptyList()
    return worldPoints.mapNotNull { worldPoint ->
        val confidence = worldPoint.confidence.coerceIn(0f, 1f)
        val anchor = createTranslationAnchor(
            session = session,
            worldPoint = worldPoint.position,
        ) ?: return@mapNotNull null
        createWorldPointMarkerAnchorNode(
            sceneView = sceneView,
            anchor = anchor,
            markerSizeMeters = SERVER_POINT_MARKER_MIN_SIZE_METERS +
                    (SERVER_POINT_MARKER_MAX_SIZE_METERS - SERVER_POINT_MARKER_MIN_SIZE_METERS) * confidence,
            materialInstance = getUnlitMaterial(
                sceneView,
                0f,
                1f,
                0f,
                confidence
            ),
        )
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
 * @param markerSizeMeters full marker size in meters.
 * @param color point marker color.
 * @return created anchor node.
 */
private fun createWorldPointMarkerAnchorNode(
    sceneView: ARSceneView,
    anchor: Anchor,
    markerSizeMeters: Float,
    materialInstance: MaterialInstance,
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
 * Creates anchor for a world-space point.
 *
 * @param session active ARCore session.
 * @param worldPoint world-space position.
 * @return created anchor or null when ARCore rejects the pose.
 */
private fun createTranslationAnchor(
    session: Session,
    worldPoint: Vector3,
): Anchor? {
    return try {
        session.createAnchor(
            Pose.makeTranslation(
                worldPoint.x,
                worldPoint.y,
                worldPoint.z,
            ),
        )
    } catch (_: FatalException) {
        null
    } catch (_: IllegalArgumentException) {
        null
    }
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
                materialInstance = getUnlitMaterial(sceneView, 0f, 0f, 1f, 1f),
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
    val edgeRotation = rotationFromPositiveXAxis(edgeDirection.normalized()) ?: return null
    val edgeAnchor = try {
        session.createAnchor(
            Pose(
                floatArrayOf(midpoint.x, midpoint.y, midpoint.z),
                floatArrayOf(edgeRotation.x, edgeRotation.y, edgeRotation.z, edgeRotation.w),
            ),
        )
    } catch (_: FatalException) {
        return null
    } catch (_: IllegalArgumentException) {
        return null
    }
    val anchorNode = AnchorNode(sceneView.engine, edgeAnchor)
    anchorNode.addChildNode(edgeNodeFactory(edgeLength))
    sceneView.addChildNode(anchorNode)
    return anchorNode
}

private fun getUnlitMaterial(sceneView: ARSceneView, r: Float, g: Float, b: Float, a: Float = 1f): MaterialInstance {
    val material = sceneView.materialLoader.createMaterial("materials/unlit_color.filamat")
    val instance = sceneView.materialLoader.createInstance(material).apply {
        setParameter("baseColor", r, g, b, a)
    }
    return instance
}

/**
 * Builds a normalized quaternion that rotates the positive X axis onto [direction].
 *
 * @param direction target unit direction.
 * @return quaternion rotating `(1, 0, 0)` onto [direction], or null when invalid.
 */
private fun rotationFromPositiveXAxis(direction: Vector3): Quaternion? {
    if (!direction.x.isFinite() || !direction.y.isFinite() || !direction.z.isFinite()) {
        return null
    }
    val source = Vector3(1f, 0f, 0f)
    val dot = source.dot(direction).coerceIn(-1f, 1f)
    val quaternion = if (dot < -1f + OPPOSITE_DIRECTION_EPSILON) {
        Quaternion(0f, 1f, 0f, 0f)
    } else {
        val cross = source.cross(direction)
        val scale = sqrt((1f + dot) * 2f)
        if (scale <= ROTATION_SCALE_EPSILON || !scale.isFinite()) {
            return null
        }
        val inverseScale = 1f / scale
        Quaternion(
            x = cross.x * inverseScale,
            y = cross.y * inverseScale,
            z = cross.z * inverseScale,
            w = scale * 0.5f,
        )
    }
    return quaternion.takeIf(::isFiniteQuaternion)?.normalized()
}

/**
 * Checks that quaternion components are all finite values.
 *
 * @param quaternion quaternion to validate.
 * @return true when all components are finite.
 */
private fun isFiniteQuaternion(quaternion: Quaternion): Boolean {
    return quaternion.x.isFinite() &&
            quaternion.y.isFinite() &&
            quaternion.z.isFinite() &&
            quaternion.w.isFinite() &&
            abs(quaternion.x) <= 1f + QUATERNION_COMPONENT_EPSILON &&
            abs(quaternion.y) <= 1f + QUATERNION_COMPONENT_EPSILON &&
            abs(quaternion.z) <= 1f + QUATERNION_COMPONENT_EPSILON &&
            abs(quaternion.w) <= 1f + QUATERNION_COMPONENT_EPSILON
}

private const val POINT_MARKER_SIZE_METERS = 0.005f
private const val SERVER_POINT_MARKER_MIN_SIZE_METERS = 0.006f
private const val SERVER_POINT_MARKER_MAX_SIZE_METERS = 0.02f
private const val LINE_THICKNESS_FACTOR = 0.02f
private const val LINE_MIN_THICKNESS_METERS = 0.003f
private const val LINE_MAX_THICKNESS_METERS = 0.03f
private const val MIN_EDGE_LENGTH_METERS = 0.001f
private const val BOUNDING_BOX_LINE_THICKNESS_METERS = 0.0015f
private const val SNAPSHOT_DIRECTION_LINE_LENGTH_METERS = 0.1f
private const val SNAPSHOT_DIRECTION_LINE_THICKNESS_METERS = 0.003f
private const val OPPOSITE_DIRECTION_EPSILON = 1e-4f
private const val ROTATION_SCALE_EPSILON = 1e-6f
private const val QUATERNION_COMPONENT_EPSILON = 1e-3f
