package anton.axenov

import io.kotest.matchers.booleans.shouldBeTrue
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.nulls.shouldNotBeNull
import io.kotest.matchers.shouldBe
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.MockRequestHandleScope
import io.ktor.client.engine.mock.respond
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.http.contentType
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.testing.ApplicationTestBuilder
import io.ktor.server.testing.testApplication
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO
import kotlin.time.Duration.Companion.milliseconds
import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3
import kotlinx.coroutines.delay
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Assumptions.assumeTrue
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
                module(fakeSegmentationPredictor())
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
     * Verifies one uploaded snapshot is processed with fake predictor and exposed in zone statuses.
     */
    @Test
    fun `snapshots endpoint should queue one fake segmentation and expose completed status`() {
        testApplication {
            resetServerState()
            application {
                module(
                    fakeSegmentationPredictor(
                        prediction = SegmentationPrediction(
                            filename = "zone-5-123.png",
                            width = 10,
                            height = 20,
                            count = 1,
                            instances = listOf(
                                SegmentationInstance(
                                    id = 1,
                                    bbox = SegmentationBoundingBox(
                                        x = 2,
                                        y = 3,
                                        width = 4,
                                        height = 5,
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            }
            val client = createJsonClient()

            val uploadResponse = client.post("/snapshots") {
                contentType(ContentType.Application.Json)
                setBody(createSnapshotPayload(screenshotPngBytes = byteArrayOf(1, 2, 3)))
            }
            val statuses = waitForZoneStatuses(client)

            uploadResponse.status shouldBe HttpStatusCode.OK
            uploadResponse.body<SnapshotUploadResponse>() shouldBe SnapshotUploadResponse(
                ok = true,
                zoneId = 5L,
                snapshotCount = 1,
                message = "stored snapshot for zone 5 and queued segmentation",
            )
            statuses shouldHaveSize 1
            statuses.single() shouldBe ZoneStatus(
                zone = 5L,
                text = "1 snapshots, queued=0, processing=0, completed=1, failed=0",
            )
        }
    }

    /**
     * Verifies one uploaded root image is segmented through the real Python predictor.
     */
    @Test
    fun `snapshots endpoint should complete one real segmentation for the known root image`() {
        assumeTrue(isPythonSegmentationServerAvailable(), "Python segmentation server must be running on 127.0.0.1:8001")

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

            val uploadResponse = client.post("/snapshots") {
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
            status.zone shouldBe 5L
            status.text.contains("completed=1").shouldBeTrue()
            status.text.contains("failed=0").shouldBeTrue()
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
        frameSnapshot = DetectionFrameSnapshotDto(
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
 * Builds fake predictor backed by Ktor mock engine.
 *
 * @param prediction segmentation response returned by fake python service.
 * @return predictor using deterministic mocked response.
 */
private fun fakeSegmentationPredictor(
    prediction: SegmentationPrediction = SegmentationPrediction(
        filename = "zone-5-123.png",
        width = 10,
        height = 20,
        count = 0,
        instances = emptyList(),
    ),
): SegmentationPredictor {
    val httpClient = HttpClient(MockEngine { request ->
        request.url.toString() shouldBe "http://fake-python/predict"
        respondJson(
            jsonBody = Json.encodeToString(SegmentationPrediction.serializer(), prediction),
        )
    }) {
        install(ContentNegotiation) {
            json(
                Json {
                    ignoreUnknownKeys = true
                },
            )
        }
    }
    return SegmentationPredictor(
        baseUrl = "http://fake-python",
        httpClient = httpClient,
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

/**
 * Checks whether the external Python segmentation server is reachable.
 *
 * @return true when localhost predictor port accepts connections.
 */
private fun isPythonSegmentationServerAvailable(): Boolean {
    return runCatching {
        Socket().use { socket ->
            socket.connect(InetSocketAddress("127.0.0.1", 8001), 500)
        }
    }.isSuccess
}

/**
 * Returns JSON mock response with standard headers.
 *
 * @param jsonBody serialized JSON body.
 * @return mock HTTP response.
 */
private fun MockRequestHandleScope.respondJson(jsonBody: String) = respond(
    content = jsonBody,
    status = HttpStatusCode.OK,
    headers = headersOf(HttpHeaders.ContentType, ContentType.Application.Json.toString()),
)
