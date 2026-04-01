package anton.axenov

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color as AndroidColor
import android.graphics.Paint
import com.google.ar.core.Anchor
import com.google.ar.core.Pose
import com.google.ar.core.Session
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.math.Color
import io.github.sceneview.node.CubeNode
import io.github.sceneview.node.ImageNode
import kotlin.math.max
import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3
import androidx.core.graphics.createBitmap
import dev.romainguy.kotlin.math.Float2

/**
 * Draws one managed zone polygon and all of its sampled points.
 *
 * @param sceneView active SceneView.
 * @param zone zone data to draw.
 * @return created anchor nodes for polygon edges, sampled points and center label.
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
    val zoneBoundingBoxNodes = createZoneBoundingBoxNodes(
        sceneView = sceneView,
        zone = zone,
    )
    val labelNode = createWorldTextLabelNode(
        sceneView = sceneView,
        worldPoint = zone.center,
        text = zone.labelText,
        normalVector = zone.planePose.normal,
    )
    return polygonNodes + pointNodes + zoneBoundingBoxNodes + listOfNotNull(labelNode)
}

/**
 * Creates a text label node in world space at provided world point and adds it to the scene.
 *
 * @param sceneView active SceneView.
 * @param worldPoint target world-space point where label anchor should be created.
 * @param text label text content.
 * @param normalVector optional world-space normal used to orient label plane.
 * @param labelHeightMeters height of label.
 * @return created label anchor node or null when session is unavailable.
 */
fun createWorldTextLabelNode(
    sceneView: ARSceneView,
    worldPoint: Vector3,
    text: String,
    normalVector: Vector3? = null,
    scale: Float = 0.5f,
): AnchorNode? {
    val session = sceneView.session ?: return null
    val textBitmap = createTextLabelBitmap(text)

    val anchor = session.createAnchor(
        createTextLabelAnchorPose(
            worldPoint = worldPoint,
            normalVector = normalVector,
        ),
    )
    val anchorNode = AnchorNode(sceneView.engine, anchor)
    val labelNode = ImageNode(
        materialLoader = sceneView.materialLoader,
        bitmap = textBitmap,
        uvScale = Float2(scale, scale)
    )
    labelNode.setCulling(false)
    anchorNode.addChildNode(labelNode)
    sceneView.addChildNode(anchorNode)
    return anchorNode
}

/**
 * Creates label anchor pose with optional orientation from the supplied plane normal.
 *
 * Local +Z axis is aligned to [normalVector] so image plane is attached to zone plane.
 *
 * @param worldPoint anchor world position.
 * @param normalVector optional zone plane normal.
 * @return ARCore pose for label anchor.
 */
private fun createTextLabelAnchorPose(
    worldPoint: Vector3,
    normalVector: Vector3?,
): Pose {
    val normalizedNormal =
        normalVector?.normalized() ?: return Pose.makeTranslation(worldPoint.x, worldPoint.y, worldPoint.z)
    val labelRotation = Quaternion.fromVectors(Vector3(0f, 0f, 1f), normalizedNormal)
    return Pose(
        floatArrayOf(worldPoint.x, worldPoint.y, worldPoint.z),
        floatArrayOf(labelRotation.x, labelRotation.y, labelRotation.z, labelRotation.w),
    )
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

/**
 * Builds simple bitmap texture with text and rounded rectangle background for AR label rendering.
 *
 * @param text text content to draw.
 * @return bitmap that can be used by SceneView `ImageNode`.
 */
private fun createTextLabelBitmap(text: String): Bitmap {
    val safeText = text.ifBlank { DEFAULT_DEBUG_LABEL_TEXT }
    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = AndroidColor.WHITE
        textSize = LABEL_TEXT_SIZE_PX
        textAlign = Paint.Align.LEFT
    }
    val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = AndroidColor.argb(220, 30, 30, 30)
        style = Paint.Style.FILL
    }
    val textWidth = textPaint.measureText(safeText).toInt()
    val width = (textWidth + LABEL_HORIZONTAL_PADDING_PX * 2).coerceAtLeast(LABEL_MIN_WIDTH_PX)
    val height = (LABEL_TEXT_SIZE_PX.toInt() + LABEL_VERTICAL_PADDING_PX * 2)
        .coerceAtLeast(LABEL_MIN_HEIGHT_PX)
    val bitmap = createBitmap(width.toInt(), height.toInt())
    val canvas = Canvas(bitmap)
    canvas.drawRoundRect(
        0f, 0f, width, height,
        LABEL_CORNER_RADIUS_PX,
        LABEL_CORNER_RADIUS_PX,
        backgroundPaint,
    )
    val baselineY = LABEL_VERTICAL_PADDING_PX + LABEL_TEXT_SIZE_PX
    canvas.drawText(
        safeText,
        LABEL_HORIZONTAL_PADDING_PX,
        baselineY,
        textPaint,
    )
    return bitmap
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
private const val LINE_THICKNESS_FACTOR = 0.02f
private const val LINE_MIN_THICKNESS_METERS = 0.003f
private const val LINE_MAX_THICKNESS_METERS = 0.03f
private const val MIN_EDGE_LENGTH_METERS = 0.001f
private const val BOUNDING_BOX_LINE_THICKNESS_METERS = 0.0015f
private const val LABEL_TEXT_SIZE_PX = 24f
private const val LABEL_HORIZONTAL_PADDING_PX = 20f
private const val LABEL_VERTICAL_PADDING_PX = 20f
private const val LABEL_CORNER_RADIUS_PX = 20f
private const val LABEL_MIN_WIDTH_PX = 300f
private const val LABEL_MIN_HEIGHT_PX = 120f
private const val DEFAULT_DEBUG_LABEL_TEXT = "Error: no text"
