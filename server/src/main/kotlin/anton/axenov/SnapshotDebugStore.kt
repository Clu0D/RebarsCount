package anton.axenov

import java.nio.file.Files
import java.nio.file.Path
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.put

/**
 * Saves debug copies of snapshot payloads without image bytes.
 *
 * @param outputDirectory directory where debug JSON payloads are stored.
 * @param json serializer used to encode payloads.
 */
class SnapshotDebugStore(
    private val outputDirectory: Path = Path.of(DEFAULT_PREDICT_POINTS_SNAPSHOT_DEBUG_DIR),
    private val json: Json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
    },
) {

    /**
     * Saves snapshot payload as JSON without image bytes.
     *
     * @param sourceImageFilename image filename that is sent to the Python predictor.
     * @param payload original uploaded snapshot payload.
     */
    fun savePredictPointsSnapshot(
        sourceImageFilename: String,
        payload: ZoneSnapshotUploadDto,
    ) {
        Files.createDirectories(outputDirectory)
        val outputPath = outputDirectory.resolve("$sourceImageFilename.json")
        Files.writeString(
            outputPath,
            json.encodeToString(
                JsonObject.serializer(),
                buildPredictPointsSnapshotDebugJson(
                    sourceImageFilename = sourceImageFilename,
                    payload = payload,
                ),
            ),
        )
    }

    private fun buildPredictPointsSnapshotDebugJson(
        sourceImageFilename: String,
        payload: ZoneSnapshotUploadDto,
    ): JsonObject {
        return buildJsonObject {
            put("sourceImageFilename", sourceImageFilename)
            put("zone", Json.encodeToJsonElement(payload.zone))
            put("frameSnapshot", buildFrameSnapshotDebugJson(payload.frameSnapshot))
            put("captureAngle", Json.encodeToJsonElement(payload.captureAngle))
            put("screenCoverage", Json.encodeToJsonElement(payload.screenCoverage))
        }
    }

    private fun buildFrameSnapshotDebugJson(
        snapshot: DetectionFrameSnapshotDto,
    ): JsonObject {
        return buildJsonObject {
            put("frameTimestamp", snapshot.frameTimestamp)
            put("imageWidth", snapshot.imageWidth)
            put("imageHeight", snapshot.imageHeight)
            put("focalLengthX", snapshot.focalLengthX)
            put("focalLengthY", snapshot.focalLengthY)
            put("principalPointX", snapshot.principalPointX)
            put("principalPointY", snapshot.principalPointY)
            put("cameraPose", Json.encodeToJsonElement(snapshot.cameraPose))
            put("depthSnapshot", Json.encodeToJsonElement(snapshot.depthSnapshot))
        }
    }

}

private const val DEFAULT_PREDICT_POINTS_SNAPSHOT_DEBUG_DIR = "server-debug/predict-points-snapshots"
