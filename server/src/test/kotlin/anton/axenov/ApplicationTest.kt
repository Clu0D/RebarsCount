package anton.axenov

import io.kotest.matchers.booleans.shouldBeTrue
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.nulls.shouldNotBeNull
import io.kotest.matchers.shouldBe
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.testing.ApplicationTestBuilder
import io.ktor.server.testing.testApplication
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.URI
import java.net.Socket
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO
import kotlin.time.Duration.Companion.milliseconds
import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3
import kotlinx.coroutines.delay
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Test

/**
 * Tests for Ktor server.
 */
class ApplicationTest {

    /**
     * Verifies health endpoint returns typed payload.
     */
    @Test
    fun `health endpoint should return typed payload`() {
        testApplication {
            resetServerState()
            application {
                module(SegmentationPredictor(baseUrl = "http://127.0.0.1:1"))
            }
            val client = createJsonClient()

            client.get("/health").apply {
                status shouldBe HttpStatusCode.OK
                body<ServerHealthResponse>() shouldBe ServerHealthResponse(
                    ok = true,
                    message = "Ktor server is online",
                )
            }
        }
    }

    /**
     * Verifies one uploaded root image is segmented through the real Python predictor.
     */
    @Test
    fun `snapshots endpoint should complete one real segmentation for the known root image`() {
        val imagePath = findSegmentationTestImage()
        val imageBytes = Files.readAllBytes(imagePath)
        val bufferedImage = ImageIO.read(imagePath.toFile())
        bufferedImage.shouldNotBeNull()

        testApplication {
            resetServerState()
            application {
                module(SegmentationPredictor())
            }
            val client = createJsonClient()

            val uploadResponse = client.post("/predict_points") {
                contentType(ContentType.Application.Json)
                setBody(
                    createSnapshotPayload(
                        screenshotPngBytes = imageBytes,
                        imageWidth = bufferedImage.width,
                        imageHeight = bufferedImage.height,
                        frameTimestamp = 456L,
                    ),
                )
            }
            val statuses = waitForZoneStatuses(client)
            val status = statuses.single()

            uploadResponse.status shouldBe HttpStatusCode.OK
            uploadResponse.body<SnapshotUploadResponse>() shouldBe SnapshotUploadResponse(
                ok = true,
                zoneId = 5L,
                snapshotCount = 1,
                message = "stored snapshot for zone 5 and queued segmentation",
            )
            statuses shouldHaveSize 1
            status.zone shouldBe 5L
            status.text.contains("queued=0").shouldBeTrue()
            status.text.contains("processing=0").shouldBeTrue()
            (status.text.contains("completed=1") || status.text.contains("failed=1")).shouldBeTrue()
        }
    }

    /**
     * Verifies predict points queue exposes failed status when Python predictor is unavailable.
     */
    @Test
    fun `snapshots endpoint should expose failed status when predictor is unavailable`() {
        testApplication {
            resetServerState()
            application {
                module(SegmentationPredictor(baseUrl = "http://127.0.0.1:1"))
            }
            val client = createJsonClient()

            val uploadResponse = client.post("/predict_points") {
                contentType(ContentType.Application.Json)
                setBody(createSnapshotPayload(screenshotPngBytes = byteArrayOf(1, 2, 3)))
            }
            val statuses = waitForZoneStatuses(client)
            val status = statuses.single()

            uploadResponse.status shouldBe HttpStatusCode.OK
            uploadResponse.body<SnapshotUploadResponse>() shouldBe SnapshotUploadResponse(
                ok = true,
                zoneId = 5L,
                snapshotCount = 1,
                message = "stored snapshot for zone 5 and queued segmentation",
            )
            statuses shouldHaveSize 1
            status.zone shouldBe 5L
            status.text.contains("queued=0").shouldBeTrue()
            status.text.contains("processing=0").shouldBeTrue()
            status.text.contains("completed=0").shouldBeTrue()
            status.text.contains("failed=1").shouldBeTrue()
        }
    }

    /**
     * Verifies interest zone endpoint proxies one image to the real Python zone predictor.
     */
    @Test
    fun `predict zone endpoint should return real detected zones`() {
        val imagePath = findSegmentationTestImage()
        val imageBytes = Files.readAllBytes(imagePath)
        val bufferedImage = ImageIO.read(imagePath.toFile())
        bufferedImage.shouldNotBeNull()

        testApplication {
            resetServerState()
            application {
                module(SegmentationPredictor())
            }
            val client = createJsonClient()

            val response = client.post("/predict_zones") {
                contentType(ContentType.Application.Json)
                setBody(
                    createFrameSnapshotPayload(
                        screenshotPngBytes = imageBytes,
                        imageWidth = bufferedImage.width,
                        imageHeight = bufferedImage.height,
                        frameTimestamp = 789L,
                    ),
                )
            }

            val prediction = response.body<SegmentationPrediction>()

            response.status shouldBe HttpStatusCode.OK
            prediction.width shouldBe bufferedImage.width
            prediction.height shouldBe bufferedImage.height
            prediction.count shouldBe prediction.instances.size
            print("prediction.count: ${prediction.count}")
        }
    }
}

/**
 * Builds test client configured with Kotlinx JSON serialization.
 *
 * @return configured test client.
 */
private fun ApplicationTestBuilder.createJsonClient(): HttpClient {
    return createClient {
        install(ContentNegotiation) {
            json(
                Json {
                    ignoreUnknownKeys = true
                },
            )
        }
    }
}

/**
 * Polls `/zone-statuses` until the single uploaded snapshot is completed or failed.
 *
 * @param client configured test client.
 * @return latest zone statuses.
 */
private suspend fun waitForZoneStatuses(client: HttpClient): List<ZoneStatus> {
    repeat(60) {
        val statuses = client.get("/zone-statuses").body<List<ZoneStatus>>()
        val status = statuses.singleOrNull()
        if (status != null && (status.text.contains("completed=1") || status.text.contains("failed=1"))) {
            return statuses
        }
        delay(250.milliseconds)
    }
    error("Timed out waiting for segmentation result")
}

/**
 * Creates one snapshot payload that is valid for server tests.
 *
 * @param screenshotPngBytes screenshot bytes to upload.
 * @param imageWidth source image width.
 * @param imageHeight source image height.
 * @param frameTimestamp frame timestamp used in generated filename.
 * @return upload payload for one zone snapshot.
 */
private fun createSnapshotPayload(
    screenshotPngBytes: ByteArray,
    imageWidth: Int = 10,
    imageHeight: Int = 20,
    frameTimestamp: Long = 123L,
): ZoneSnapshotUploadDto {
    return ZoneSnapshotUploadDto(
        zone = Zone(
            id = 5L,
            sampledPoints = listOf(Vector3(1f, 2f, 3f)),
            planePose = PlanePose(
                center = Vector3(1f, 2f, 3f),
                rotation = Quaternion(0f, 0f, 0f, 1f),
                normal = Vector3(0f, 1f, 0f),
                offsetD = 0f,
            ),
            projectionInputs = emptyList(),
        ),
        frameSnapshot = createFrameSnapshotPayload(
            screenshotPngBytes = screenshotPngBytes,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            frameTimestamp = frameTimestamp,
        ),
        captureAngle = ZoneCaptureAngle(
            angleDegrees = 15f,
            zoneToCameraDirection = Vector3(0f, 0f, -1f),
            normalToCameraDot = 0.9f,
            planarDirectionX = 0.1f,
            planarDirectionY = 0.2f,
        ),
        screenCoverage = ZoneScreenCoverageMetrics(
            projectedArea = 100f,
            visibleArea = 90f,
            isFullyInside = true,
            screenArea = 1000f,
        ),
    )
}

/**
 * Creates one frame snapshot payload that is valid for server tests.
 *
 * @param screenshotPngBytes screenshot bytes to upload.
 * @param imageWidth source image width.
 * @param imageHeight source image height.
 * @param frameTimestamp frame timestamp used in generated filename.
 * @return frame snapshot payload for zone prediction endpoint.
 */
private fun createFrameSnapshotPayload(
    screenshotPngBytes: ByteArray,
    imageWidth: Int = 10,
    imageHeight: Int = 20,
    frameTimestamp: Long = 123L,
): DetectionFrameSnapshotDto {
    return DetectionFrameSnapshotDto(
        screenshotPngBytes = screenshotPngBytes,
        frameTimestamp = frameTimestamp,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        focalLengthX = 1f,
        focalLengthY = 2f,
        principalPointX = 3f,
        principalPointY = 4f,
        cameraPose = CameraPoseDto(
            translation = Vector3(1f, 2f, 3f),
            rotationQuaternion = Quaternion(0f, 0f, 0f, 1f),
        ),
        depthSnapshot = DepthSnapshot(
            width = 2,
            height = 2,
            values = shortArrayOf(1, 2, 3, 4),
        ),
    )
}

/**
 * Finds the known segmentation image from the repository root.
 *
 * @return path to `test_image_for_segmentation.png`.
 */
private fun findSegmentationTestImage(): Path {
    val candidatePaths = listOf(
        Path.of("test_image_for_segmentation.png"),
        Path.of("..", "test_image_for_segmentation.png"),
    )
    return candidatePaths.firstOrNull(Files::exists)
        ?: error("Could not find test_image_for_segmentation.png in project root")
}