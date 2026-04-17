@file:UseSerializers(Vector3Serializer::class, QuaternionSerializer::class)

package anton.axenov

import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3
import kotlin.math.max
import kotlin.math.min
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.UseSerializers
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.descriptors.element
import kotlinx.serialization.encoding.CompositeDecoder
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure

/**
 * Plane pose payload returned by shared fitting logic.
 *
 * @param center plane center.
 * @param rotation orientation quaternion where local +Z is the plane normal.
 * @param normal fitted normal.
 * @param offsetD plane offset in `ax + by + cz + d = 0`.
 */
@Serializable
data class PlanePose(
    val center: Vector3,
    val rotation: Quaternion,
    val normal: Vector3,
    val offsetD: Float = -normal.dot(center),
)

/**
 * Pixel point in image coordinates.
 *
 * @param x X coordinate.
 * @param y Y coordinate.
 */
@Serializable
data class ImagePoint(
    val x: Int,
    val y: Int,
)

/**
 * Translation strategy by orientation.
 */
@Serializable
enum class CoordinateTranslationVariant {
    PORTRAIT,
    LANDSCAPE,
    LANDSCAPE_REVERSED,
}

/**
 * Translates one detector-space point to image-space coordinates used by placement math.
 *
 * @param x detector-space X coordinate.
 * @param y detector-space Y coordinate.
 * @param width source image width in pixels.
 * @param height source image height in pixels.
 * @param translationVariant detector-to-image translation variant.
 * @return translated and clamped image-space point.
 */
fun translateCoordinates(
    x: Int,
    y: Int,
    width: Int,
    height: Int,
    translationVariant: CoordinateTranslationVariant,
): ImagePoint {
    if (width <= 0 || height <= 0) {
        return ImagePoint(0, 0)
    }

    val safeX = x.coerceIn(0, width - 1)
    val safeY = y.coerceIn(0, height - 1)
    return when (translationVariant) {
        CoordinateTranslationVariant.PORTRAIT -> ImagePoint(
            x = safeY * width / height,
            y = (width - 1 - safeX) * height / width,
        )

        CoordinateTranslationVariant.LANDSCAPE -> ImagePoint(
            x = safeX,
            y = safeY,
        )

        CoordinateTranslationVariant.LANDSCAPE_REVERSED -> ImagePoint(
            x = width - 1 - safeX,
            y = height - 1 - safeY,
        )
    }
}

/**
 * Full projection payload required to reproject one zone polygon onto any plane.
 *
 * @param originalScreenPolygon detector-space polygon points before translation.
 * @param translationVariant detector-to-image translation variant.
 * @param imageWidth captured image width.
 * @param imageHeight captured image height.
 * @param focalLengthX camera focal length X in pixels.
 * @param focalLengthY camera focal length Y in pixels.
 * @param principalPointX camera principal point X in pixels.
 * @param principalPointY camera principal point Y in pixels.
 * @param cameraToWorldMatrix camera pose matrix in OpenGL column-major form.
 */
@Serializable
class ZoneProjectionInput(
    val originalScreenPolygon: List<ImagePoint>,
    val translationVariant: CoordinateTranslationVariant,
    val imageWidth: Int,
    val imageHeight: Int,
    val focalLengthX: Float,
    val focalLengthY: Float,
    val principalPointX: Float,
    val principalPointY: Float,
    val cameraToWorldMatrix: FloatArray,
) {
    /**
     * Returns camera world position extracted from [cameraToWorldMatrix].
     *
     * @return camera world position.
     */
    fun cameraPosition(): Vector3 {
        return Vector3(
            x = cameraToWorldMatrix[12],
            y = cameraToWorldMatrix[13],
            z = cameraToWorldMatrix[14],
        )
    }

    /**
     * Projects original on-screen polygon onto [planePose].
     *
     * @param planePose projection plane.
     * @return projected world-space polygon or null when at least one corner cannot be projected.
     */
    fun projectToPlane(planePose: PlanePose): List<Vector3>? {
        val translatedCorners = originalScreenPolygon.map { corner ->
            translateCoordinates(
                x = corner.x,
                y = corner.y,
                width = imageWidth,
                height = imageHeight,
                translationVariant = translationVariant,
            )
        }
        val projectedCorners = translatedCorners.map { corner ->
            projectImagePointToPlane(
                imagePoint = corner,
                planePose = planePose,
            )
        }
        if (projectedCorners.any { it == null }) {
            return null
        }
        return projectedCorners.map { it!! }
    }

    /**
     * Projects one image-space point to [planePose].
     *
     * @param imagePoint image-space point to project.
     * @param planePose projection plane.
     * @return world-space intersection or null when ray is parallel to plane or intersects behind camera.
     */
    private fun projectImagePointToPlane(
        imagePoint: ImagePoint,
        planePose: PlanePose,
    ): Vector3? {
        if (
            imageWidth <= 0 ||
            imageHeight <= 0 ||
            focalLengthX == 0f ||
            focalLengthY == 0f
        ) {
            return null
        }
        val imageX = imagePoint.x.coerceIn(0, imageWidth - 1)
        val imageY = imagePoint.y.coerceIn(0, imageHeight - 1)

        val directionCamera = Vector3(
            x = (imageX - principalPointX) / focalLengthX,
            y = -(imageY - principalPointY) / focalLengthY,
            z = -1f,
        ).normalized()
        val matrix = cameraToWorldMatrix
        val rayOrigin = cameraPosition()
        val rayDirection = Vector3(
            x = matrix[0] * directionCamera.x + matrix[4] * directionCamera.y + matrix[8] * directionCamera.z,
            y = matrix[1] * directionCamera.x + matrix[5] * directionCamera.y + matrix[9] * directionCamera.z,
            z = matrix[2] * directionCamera.x + matrix[6] * directionCamera.y + matrix[10] * directionCamera.z,
        ).normalized()

        val denominator = planePose.normal.dot(rayDirection)
        if (kotlin.math.abs(denominator) <= RAY_PLANE_EPSILON) {
            return null
        }
        val distance = planePose.normal.dot(planePose.center - rayOrigin) / denominator
        if (distance <= 0f) {
            return null
        }
        return rayOrigin + rayDirection * distance
    }
}

/**
 * One detected zone represented in world space.
 *
 * @param id unique zone identifier.
 * @param sampledPoints sampled world points used to estimate infinite plane.
 * @param planePose mathematical parameters of fitted infinite plane.
 * @param projectionInputs all original projection payloads used to build source polygons.
 * @param insignificantChanges number of last changes that are small enough to place a zone.
 */
@Serializable
data class Zone(
    val id: Long = nextZoneId(),
    val sampledPoints: List<Vector3>,
    val planePose: PlanePose,
    val projectionInputs: List<ZoneProjectionInput> = emptyList(),
    var insignificantChanges: Int = 0,
) {
    var metricsLabelText: String = "Metrics: waiting"
    var serverLabelText: String? = null
    var mergeLabelText: String? = null
    var sceneWorldPointsCount: Int = 0

    /**
     * Is true when the zone has been placed in the world and can be used for segmentation.
     */
    fun isPlaced(): Boolean = insignificantChanges >= INSIGNIFICANT_CHANGES_BEFORE_PLACE_ZONE

    /**
     * Current text shown in the zone label in AR scene.
     */
    val labelText: String
        get() = "$metricsLabelText\n${serverLabelText ?: ""}"

    val polygonPoints: List<Vector3> by lazy {
        if (projectionInputs.isEmpty()) {
            return@lazy emptyList()
        }

        val projectedPolygons = projectionInputs
            .mapNotNull { input -> input.projectToPlane(planePose) }
        if (projectedPolygons.isEmpty()) {
            return@lazy emptyList()
        }

        buildConfidenceConvexHullOnPlane(
            worldPoints = projectedPolygons,
            planePose = planePose,
        )
    }

    val boundingBox: ZoneBoundingBox3d by lazy {
        val basePoints = when {
            polygonPoints.isNotEmpty() -> polygonPoints
            sampledPoints.isNotEmpty() -> sampledPoints
            else -> listOf(planePose.center)
        }

        val minX = basePoints.minOf { it.x }
        val minY = basePoints.minOf { it.y }
        val minZ = basePoints.minOf { it.z }
        val maxX = basePoints.maxOf { it.x }
        val maxY = basePoints.maxOf { it.y }
        val maxZ = basePoints.maxOf { it.z }

        val sizeX = maxX - minX
        val sizeY = maxY - minY
        val sizeZ = maxZ - minZ
        val maxSize = max(max(sizeX, sizeY), sizeZ)
        val padding = max(maxSize * BOUNDING_BOX_PADDING_RATIO, MIN_PADDING_METERS)

        ZoneBoundingBox3d(
            minX = minX - padding,
            minY = minY - padding,
            minZ = minZ - padding,
            maxX = maxX + padding,
            maxY = maxY + padding,
            maxZ = maxZ + padding,
        )
    }

    val center: Vector3 by lazy {
        Vector3(
            (boundingBox.maxX + boundingBox.minX) / 2f,
            (boundingBox.maxY + boundingBox.minY) / 2f,
            (boundingBox.maxZ + boundingBox.minZ) / 2f,
        )
    }
}

/**
 * Axis-aligned 3D bounds used to compare zone overlap.
 *
 * @param minX minimum X coordinate.
 * @param minY minimum Y coordinate.
 * @param minZ minimum Z coordinate.
 * @param maxX maximum X coordinate.
 * @param maxY maximum Y coordinate.
 * @param maxZ maximum Z coordinate.
 */
data class ZoneBoundingBox3d(
    val minX: Float,
    val minY: Float,
    val minZ: Float,
    val maxX: Float,
    val maxY: Float,
    val maxZ: Float,
) {
    /**
     * Computes this box volume.
     *
     * @return positive box volume.
     */
    fun volume(): Float {
        return (maxX - minX).coerceAtLeast(0f) *
                (maxY - minY).coerceAtLeast(0f) *
                (maxZ - minZ).coerceAtLeast(0f)
    }

    /**
     * Computes intersection volume with another box.
     *
     * @param other second box.
     * @return intersection volume or zero when there is no overlap.
     */
    fun intersectionVolume(other: ZoneBoundingBox3d): Float {
        val overlapX = min(maxX, other.maxX) - max(minX, other.minX)
        val overlapY = min(maxY, other.maxY) - max(minY, other.minY)
        val overlapZ = min(maxZ, other.maxZ) - max(minZ, other.minZ)
        if (overlapX <= 0f || overlapY <= 0f || overlapZ <= 0f) {
            return 0f
        }
        return overlapX * overlapY * overlapZ
    }

    /**
     * Computes overlap ratio relative to the smaller box volume.
     *
     * @param other second box.
     * @return value in `[0, 1]` where `0` means no overlap.
     */
    fun overlapRatioBySmallerBox(other: ZoneBoundingBox3d): Float {
        val intersection = intersectionVolume(other)
        if (intersection <= 0f) {
            return 0f
        }
        val denominator = min(volume(), other.volume()).coerceAtLeast(MIN_BOX_VOLUME_EPSILON)
        return intersection / denominator
    }
}

/**
 * Camera-to-zone orientation metrics for one screenshot.
 *
 * @param angleDegrees angle between zone normal and zone-to-camera direction in degrees.
 * @param zoneToCameraDirection normalized direction from plane center to camera position.
 * @param normalToCameraDot dot product of normalized plane normal and zone-to-camera direction.
 * @param planarDirectionX direction to the camera projected onto zone plane local 2D axes by X.
 * @param planarDirectionY direction to the camera projected onto zone plane local 2D axes by Y.
 */
@Serializable
data class ZoneCaptureAngle(
    val angleDegrees: Float,
    val zoneToCameraDirection: Vector3,
    val normalToCameraDot: Float,
    val planarDirectionX: Float,
    val planarDirectionY: Float,
) {
    /**
     * Calculates angle between this capture direction and another capture direction.
     *
     * @param other another capture direction.
     * @return angle in degrees in range [0, 180].
     */
    fun sphericalAngleTo(other: ZoneCaptureAngle): Float {
        val firstNormalized = zoneToCameraDirection.normalized()
        val secondNormalized = other.zoneToCameraDirection.normalized()
        val dot = firstNormalized.dot(secondNormalized).coerceIn(-1f, 1f)
        return (kotlin.math.acos(dot.toDouble()) * 180.0 / kotlin.math.PI).toFloat()
    }
}

/**
 * Screen coverage metrics for one projected zone polygon.
 *
 * @param projectedArea full projected polygon area in pixels.
 * @param visibleArea polygon area clipped to screen rectangle in pixels.
 * @param isFullyInside true when polygon lies inside screen bounds.
 * @param screenArea full screen area in pixels.
 * @param coverage visible polygon area relative to full screen area.
 */
@Serializable
data class ZoneScreenCoverageMetrics(
    val projectedArea: Float,
    val visibleArea: Float,
    val isFullyInside: Boolean,
    val screenArea: Float,
    val coverage: Float = if (screenArea <= 0f) 0f else visibleArea / screenArea,
)

/**
 * Depth map sampled from one frame.
 *
 * Values are raw ARCore `DEPTH16` values.
 *
 * @param width depth image width.
 * @param height depth image height.
 * @param values depth values in row-major order.
 */
@Serializable(with = DepthSnapshotSerializer::class)
data class DepthSnapshot(
    val width: Int,
    val height: Int,
    val values: ShortArray,
)

/**
 * Serializes [DepthSnapshot] while preserving raw unsigned `DEPTH16` values.
 */
object DepthSnapshotSerializer : KSerializer<DepthSnapshot> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("DepthSnapshot") {
        element<Int>("width")
        element<Int>("height")
        element("values", ListSerializer(Int.serializer()).descriptor)
    }

    override fun serialize(encoder: Encoder, value: DepthSnapshot) {
        encoder.encodeStructure(descriptor) {
            encodeIntElement(descriptor, 0, value.width)
            encodeIntElement(descriptor, 1, value.height)
            encodeSerializableElement(
                descriptor,
                2,
                ListSerializer(Int.serializer()),
                value.values.map { depth -> depth.toInt() and 0xFFFF },
            )
        }
    }

    override fun deserialize(decoder: Decoder): DepthSnapshot {
        var width = 0
        var height = 0
        var values = emptyList<Int>()
        decoder.decodeStructure(descriptor) {
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break
                    0 -> width = decodeIntElement(descriptor, index)
                    1 -> height = decodeIntElement(descriptor, index)
                    2 -> values = decodeSerializableElement(
                        descriptor,
                        index,
                        ListSerializer(Int.serializer()),
                    )
                    else -> error("Unexpected depth snapshot element index: $index")
                }
            }
        }
        return DepthSnapshot(
            width = width,
            height = height,
            values = values.map { depth -> depth.toShort() }.toShortArray(),
        )
    }
}

private const val INSIGNIFICANT_CHANGES_BEFORE_PLACE_ZONE = 3
private const val BOUNDING_BOX_PADDING_RATIO = 0.1f
private const val MIN_PADDING_METERS = 0.05f
private const val MIN_BOX_VOLUME_EPSILON = 1e-8f
private const val RAY_PLANE_EPSILON = 1e-5f

private var nextGeneratedZoneId: Long = 1L
private fun nextZoneId(): Long = nextGeneratedZoneId++
