package anton.axenov

import com.google.ar.core.Anchor
import com.google.ar.core.Pose
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.node.CubeNode
import korlibs.math.geom.Vector3F as Vector3

/**
 * Creates a rectangle marker anchor node and adds it to the scene.
 *
 * @param sceneView active SceneView.
 * @param anchor anchor to attach marker node to.
 * @param rectangleWidthMeters rectangle width in world meters.
 * @param rectangleHeightMeters rectangle height in world meters.
 * @return created anchor node.
 */
fun createRectangleMarkerAnchorNode(
    sceneView: ARSceneView,
    anchor: Anchor,
    rectangleWidthMeters: Float,
    rectangleHeightMeters: Float,
): AnchorNode {
    val anchorNode = AnchorNode(sceneView.engine, anchor)
    val halfWidth = rectangleWidthMeters / 2f
    val halfHeight = rectangleHeightMeters / 2f
    val lineThickness = (rectangleWidthMeters.coerceAtMost(rectangleHeightMeters) * RECTANGLE_LINE_THICKNESS_FACTOR)
        .coerceAtLeast(RECTANGLE_LINE_MIN_THICKNESS_METERS)
        .coerceAtMost(RECTANGLE_LINE_MAX_THICKNESS_METERS)

    val topEdge = CubeNode(
        engine = sceneView.engine,
        size = dev.romainguy.kotlin.math.Float3(rectangleWidthMeters, lineThickness, lineThickness),
    )
    topEdge.position = dev.romainguy.kotlin.math.Float3(0f, halfHeight, 0f)
    val bottomEdge = CubeNode(
        engine = sceneView.engine,
        size = dev.romainguy.kotlin.math.Float3(rectangleWidthMeters, lineThickness, lineThickness),
    )
    bottomEdge.position = dev.romainguy.kotlin.math.Float3(0f, -halfHeight, 0f)
    val leftEdge = CubeNode(
        engine = sceneView.engine,
        size = dev.romainguy.kotlin.math.Float3(lineThickness, rectangleHeightMeters, lineThickness),
    )
    leftEdge.position = dev.romainguy.kotlin.math.Float3(-halfWidth, 0f, 0f)
    val rightEdge = CubeNode(
        engine = sceneView.engine,
        size = dev.romainguy.kotlin.math.Float3(lineThickness, rectangleHeightMeters, lineThickness),
    )
    rightEdge.position = dev.romainguy.kotlin.math.Float3(halfWidth, 0f, 0f)

    anchorNode.addChildNode(topEdge)
    anchorNode.addChildNode(bottomEdge)
    anchorNode.addChildNode(leftEdge)
    anchorNode.addChildNode(rightEdge)
    sceneView.addChildNode(anchorNode)
    return anchorNode
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
private const val RECTANGLE_LINE_THICKNESS_FACTOR = 0.02f
private const val RECTANGLE_LINE_MIN_THICKNESS_METERS = 0.003f
private const val RECTANGLE_LINE_MAX_THICKNESS_METERS = 0.03f
