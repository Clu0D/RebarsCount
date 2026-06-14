package anton.axenov

import kotlin.test.Test
import io.kotest.matchers.booleans.shouldBeFalse
import io.kotest.matchers.booleans.shouldBeTrue
import io.kotest.matchers.shouldBe
import korlibs.math.geom.Quaternion
import korlibs.math.geom.Vector3F
import kotlinx.coroutines.runBlocking

/**
 * Tests mutable world-point post-processing in [SegmentationSessionProcessor].
 */
class SegmentationSessionProcessorTest {
    @Test
    fun `added point should use nearest zone and can be deleted by stable id`() {
        runBlocking {
            val processor = postProcessingProcessor()
            try {
                registerPostProcessingZones(processor, listOf(0f, 10f))

                val added = processor.addWorldPoint(
                    AddWorldPointDto(position = Vector3F(8f, 0f, 0f)),
                )
                added.ok.shouldBeTrue()
                added.point?.zoneId shouldBe 2L
                val pointId = added.point!!.pointId
                processor.fetchWorldPoints().single().pointId shouldBe pointId

                val deleted = processor.deleteWorldPoint(pointId)
                deleted.ok.shouldBeTrue()
                processor.fetchWorldPoints() shouldBe emptyList()
                processor.deleteWorldPoint(pointId).ok.shouldBeFalse()
            } finally {
                processor.close()
            }
        }
    }

    @Test
    fun `zone rotation should cycle through only four nearest zone centers`() {
        runBlocking {
            val processor = postProcessingProcessor()
            try {
                registerPostProcessingZones(processor, listOf(0f, 10f, 20f, 30f, 40f))
                val pointId = processor.addWorldPoint(
                    AddWorldPointDto(
                        position = Vector3F(1f, 0f, 0f),
                        zoneId = 1L,
                    ),
                ).point!!.pointId

                val rotatedZoneIds = List(5) {
                    processor.rotateWorldPointZone(pointId).point!!.zoneId
                }

                rotatedZoneIds shouldBe listOf(2L, 3L, 4L, 1L, 2L)
            } finally {
                processor.close()
            }
        }
    }

    @Test
    fun `zone deletion should remove snapshots queue and related points`() {
        runBlocking {
            val processor = postProcessingProcessor()
            try {
                registerPostProcessingZones(processor, listOf(0f, 10f))
                processor.addWorldPoint(
                    AddWorldPointDto(position = Vector3F(1f, 0f, 0f), zoneId = 1L),
                )
                val retainedPoint = processor.addWorldPoint(
                    AddWorldPointDto(position = Vector3F(9f, 0f, 0f), zoneId = 2L),
                ).point!!

                val deletion = processor.deleteZone(1L)

                deletion.ok.shouldBeTrue()
                deletion.removedSnapshots shouldBe 1
                deletion.removedQueuedTasks shouldBe 1
                deletion.removedWorldPoints shouldBe 1
                processor.fetchZoneStatuses().map { status -> status.zone } shouldBe listOf(2L)
                processor.fetchWorldPoints() shouldBe listOf(retainedPoint)
                processor.deleteZone(1L).ok.shouldBeFalse()
            } finally {
                processor.close()
            }
        }
    }
}

/**
 * Creates an isolated processor without background workers.
 *
 * @return processor suitable for post-processing tests.
 */
private fun postProcessingProcessor(): SegmentationSessionProcessor {
    return SegmentationSessionProcessor(
        predictor = EmptyPostProcessingPredictionProvider(),
        sessionId = POST_PROCESSING_SESSION_ID,
        workerCount = 0,
    )
}

/**
 * Registers zones by submitting one queued snapshot for each center.
 *
 * @param processor processor that should receive zones.
 * @param centerXs X coordinates of zones in identifier order.
 */
private suspend fun registerPostProcessingZones(
    processor: SegmentationSessionProcessor,
    centerXs: List<Float>,
) {
    centerXs.forEachIndexed { index, centerX ->
        processor.predictPoints(postProcessingPayload(zoneId = index + 1L, centerX = centerX))
    }
}

/**
 * Builds one minimal zone snapshot payload.
 *
 * @param zoneId zone identifier.
 * @param centerX zone center X coordinate.
 * @return valid snapshot upload.
 */
private fun postProcessingPayload(zoneId: Long, centerX: Float): ZoneSnapshotUploadDto {
    val center = Vector3F(centerX, 0f, 0f)
    return ZoneSnapshotUploadDto(
        sessionId = POST_PROCESSING_SESSION_ID,
        requestId = "post-processing-$zoneId",
        zone = Zone(
            id = zoneId,
            sampledPoints = listOf(center),
            planePose = PlanePose(
                center = center,
                rotation = Quaternion.IDENTITY,
                normal = Vector3F(0f, 0f, 1f),
            ),
        ),
        frameSnapshot = DetectionFrameSnapshotDto(
            screenshotPngBytes = byteArrayOf(1),
            frameTimestamp = zoneId,
            imageWidth = 1,
            imageHeight = 1,
            focalLengthX = 1f,
            focalLengthY = 1f,
            principalPointX = 0f,
            principalPointY = 0f,
            cameraPose = CameraPoseDto(
                translation = Vector3F(0f, 0f, 0f),
                rotationQuaternion = Quaternion.IDENTITY,
            ),
            depthSnapshot = null,
        ),
        captureAngle = ZoneCaptureAngle(
            angleDegrees = 0f,
            zoneToCameraDirection = Vector3F(0f, 0f, 1f),
            normalToCameraDot = 1f,
            planarDirectionX = 0f,
            planarDirectionY = 0f,
        ),
        screenCoverage = ZoneScreenCoverageMetrics(
            projectedArea = 1f,
            visibleArea = 1f,
            isFullyInside = true,
            screenArea = 1f,
        ),
    )
}

/**
 * Prediction provider unused by post-processing tests.
 */
private class EmptyPostProcessingPredictionProvider : SegmentationPredictionProvider {
    /**
     * Returns an empty prediction.
     *
     * @param imageBytes ignored image bytes.
     * @param filename logical filename.
     * @param zonePrediction ignored prediction kind.
     * @return empty prediction.
     */
    override suspend fun predict(
        imageBytes: ByteArray,
        filename: String,
        zonePrediction: Boolean,
    ): SegmentationPrediction {
        return SegmentationPrediction(
            filename = filename,
            width = 1,
            height = 1,
            count = 0,
            instances = emptyList(),
        )
    }

    /**
     * Releases no resources.
     */
    override fun close() = Unit
}

private const val POST_PROCESSING_SESSION_ID = "post-processing-test"
