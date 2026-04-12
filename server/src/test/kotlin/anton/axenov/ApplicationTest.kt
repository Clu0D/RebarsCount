package anton.axenov

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
import kotlin.test.Test
import kotlin.test.assertEquals
import korlibs.math.geom.Quaternion as Quaternion
import korlibs.math.geom.Vector3F as Vector3
import kotlinx.serialization.json.Json

/**
 * Tests for Ktor server routes.
 */
class ApplicationTest {

    /**
     * Verifies health endpoint returns typed JSON payload.
     */
    @Test
    fun testHealth() = testApplication {
        resetServerState()
        application {
            module()
        }
        val client = createJsonClient()

        val response = client.get("/health")

        assertEquals(HttpStatusCode.OK, response.status)
        assertEquals(
            ServerHealthResponseDto(ok = true, message = "Ktor server is online"),
            response.body(),
        )
    }

    /**
     * Verifies one uploaded snapshot is reflected in statuses and stored snapshots route.
     */
    @Test
    fun testSnapshotFlow() = testApplication {
        resetServerState()
        application {
            module()
        }
        val client = createJsonClient()
        val payload = ZoneSnapshotUploadDto(
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
                screenshotPngBytes = byteArrayOf(1, 2, 3),
                frameTimestamp = 123L,
                imageWidth = 10,
                imageHeight = 20,
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

        val uploadResponse = client.post("/snapshots") {
            contentType(ContentType.Application.Json)
            setBody(payload)
        }
        val statusesResponse = client.get("/zone-statuses")
        val snapshotsResponse = client.get("/snapshots/5")

        assertEquals(HttpStatusCode.OK, uploadResponse.status)
        assertEquals(
            SnapshotUploadResponseDto(
                ok = true,
                zoneId = 5L,
                snapshotCount = 1,
                message = "stored snapshot for zone 5",
            ),
            uploadResponse.body(),
        )
        assertEquals(
            listOf(ZoneStatusDto(zone = 5L, text = "1 snapshot(s) uploaded")),
            statusesResponse.body(),
        )
        val snapshotsBody = snapshotsResponse.body<ZoneSnapshotsResponseDto>()
        assertEquals(5L, snapshotsBody.zoneId)
        assertEquals(1, snapshotsBody.snapshotCount)
        assertEquals("1 snapshot(s) uploaded", snapshotsBody.text)
        assertEquals(1, snapshotsBody.snapshots.size)
        assertEquals(5L, snapshotsBody.snapshots.single().zone.id)
        assertEquals(123L, snapshotsBody.snapshots.single().frameSnapshot.frameTimestamp)
    }

    /**
     * Builds test client configured with Kotlinx JSON serialization.
     *
     * @return configured test client.
     */
    private fun ApplicationTestBuilder.createJsonClient() = createClient {
        install(ContentNegotiation) {
            json(
                Json {
                    ignoreUnknownKeys = true
                },
            )
        }
    }
}
