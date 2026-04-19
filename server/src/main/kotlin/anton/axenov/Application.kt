package anton.axenov

import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.application.log
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import io.ktor.serialization.kotlinx.KotlinxSerializationConverter
import io.ktor.http.ContentType
import kotlinx.serialization.json.Json

private val segmentationQueue = SegmentationQueue()
private val serverJson = Json {
    ignoreUnknownKeys = true
    prettyPrint = true
}

/**
 * Clears in-memory server state.
 */
fun resetServerState() {
    segmentationQueue.clear()
}

fun main() {
    embeddedServer(Netty, port = SERVER_PORT, host = "0.0.0.0", module = Application::module)
        .start(wait = true)
}

/**
 * Configures HTTP routes for the AR snapshot server.
 */
fun Application.module(
    predictor: SegmentationPredictor = SegmentationPredictor(),
) {
    segmentationQueue.configure(predictor)
    val jsonConverter = KotlinxSerializationConverter(serverJson)
    install(ContentNegotiation) {
        register(ContentType.Application.Json, jsonConverter)
        register(ContentType.Text.Plain, jsonConverter)
        register(ContentType.Application.OctetStream, jsonConverter)
    }

    routing {
        get("/") {
            call.respond(ServerHealthResponse(ok = true, message = "Ktor server is online"))
        }

        get("/health") {
            call.respond(ServerHealthResponse(ok = true, message = "Ktor server is online"))
        }

        post("/predict_points") {
            val payload = call.receive<ZoneSnapshotUploadDto>()
            val snapshotCount = segmentationQueue.addSnapshot(payload)
            call.respond(
                SnapshotUploadResponse(
                    ok = true,
                    zoneId = payload.zone.id,
                    snapshotCount = snapshotCount,
                    message = "stored snapshot for zone ${payload.zone.id} and queued segmentation",
                ),
            )
        }

        post("/predict_zones") {
            val payload = call.receive<DetectionFrameSnapshotDto>()
            val prediction = predictor.predict(
                imageBytes = payload.screenshotPngBytes,
                filename = "${payload.frameTimestamp}-zones-seg.png",
                zonePrediction = true
            )
            call.respond(prediction)
        }

        get("/zone-statuses") {
            call.respond(segmentationQueue.getZoneStatuses())
        }

        get("/world-points") {
            call.respond(segmentationQueue.getAllWorldPoints())
        }

        post("/start_new_session") {
            resetServerState()
            call.respond(
                ServerHealthResponse(
                    ok = true,
                    message = "Started new session and cleared server state",
                ),
            )
        }
    }

    monitor.subscribe(io.ktor.server.application.ApplicationStopped) {
        log.info("Stopping segmentation queue")
        segmentationQueue.stop()
    }
}