package anton.axenov

import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.descriptors.element
import kotlinx.serialization.encoding.CompositeDecoder
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure

/**
 * Serializes korlibs Vector3F as JSON object with `x`, `y` and `z`.
 */
object Vector3Serializer : KSerializer<Vector3F> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("Vector3") {
        element<Float>("x")
        element<Float>("y")
        element<Float>("z")
    }

    override fun serialize(encoder: Encoder, value: Vector3F) {
        encoder.encodeStructure(descriptor) {
            encodeFloatElement(descriptor, 0, value.x)
            encodeFloatElement(descriptor, 1, value.y)
            encodeFloatElement(descriptor, 2, value.z)
        }
    }

    override fun deserialize(decoder: Decoder): Vector3F {
        var x = 0f
        var y = 0f
        var z = 0f
        decoder.decodeStructure(descriptor) {
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break
                    0 -> x = decodeFloatElement(descriptor, index)
                    1 -> y = decodeFloatElement(descriptor, index)
                    2 -> z = decodeFloatElement(descriptor, index)
                    else -> error("Unexpected vector element index: $index")
                }
            }
        }
        return Vector3F(x, y, z)
    }
}

/**
 * Serializes korlibs [Quaternion] as JSON object with `x`, `y`, `z` and `w`.
 */
object QuaternionSerializer : KSerializer<Quaternion> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("Quaternion") {
        element<Float>("x")
        element<Float>("y")
        element<Float>("z")
        element<Float>("w")
    }

    override fun serialize(encoder: Encoder, value: Quaternion) {
        encoder.encodeStructure(descriptor) {
            encodeFloatElement(descriptor, 0, value.x)
            encodeFloatElement(descriptor, 1, value.y)
            encodeFloatElement(descriptor, 2, value.z)
            encodeFloatElement(descriptor, 3, value.w)
        }
    }

    override fun deserialize(decoder: Decoder): Quaternion {
        var x = 0f
        var y = 0f
        var z = 0f
        var w = 1f
        decoder.decodeStructure(descriptor) {
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    CompositeDecoder.DECODE_DONE -> break
                    0 -> x = decodeFloatElement(descriptor, index)
                    1 -> y = decodeFloatElement(descriptor, index)
                    2 -> z = decodeFloatElement(descriptor, index)
                    3 -> w = decodeFloatElement(descriptor, index)
                    else -> error("Unexpected quaternion element index: $index")
                }
            }
        }
        return Quaternion(x, y, z, w)
    }
}
