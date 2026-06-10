package anton.axenov

import anton.axenov.SegmentationSessionProcessor
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
import io.ktor.http.HttpStatusCode
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json

private var segmentationProcessor = SegmentationSessionProcessor(
    predictor = SegmentationPredictor(),
    onSnapshotAccepted = SnapshotDebugStore()::savePredictPointsSnapshot,
)
private val serverJson = Json {
    ignoreUnknownKeys = true
    prettyPrint = true
}

/**
 * Clears in-memory server state.
 */
fun resetServerState() {
    runBlocking {
        segmentationProcessor.clear()
    }
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
    segmentationProcessor.close()
    segmentationProcessor = SegmentationSessionProcessor(
        predictor = predictor,
        onSnapshotAccepted = SnapshotDebugStore()::savePredictPointsSnapshot,
    )
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
            call.respond(segmentationProcessor.predictPoints(payload))
        }

        post("/predict_zones") {
            val payload = call.receive<DetectionFrameSnapshotDto>()
            val prediction = try {
                segmentationProcessor.predictZones(payload)
            } catch (error: Exception) {
                if (error is CancellationException) {
                    throw error
                }
                log.error("Zone prediction failed", error)
                call.respond(
                    HttpStatusCode.BadGateway,
                    ServerHealthResponse(
                        ok = false,
                        message = error.message ?: "Unknown Python segmentation error",
                    ),
                )
                return@post
            }
            call.respond(prediction)
        }

        get("/zone-statuses") {
            call.respond(segmentationProcessor.fetchZoneStatuses())
        }

        get("/world-points") {
            call.respond(segmentationProcessor.fetchWorldPoints())
        }

        post("/start_new_session") {
            call.respond(segmentationProcessor.startNewSession())
        }
    }

    monitor.subscribe(io.ktor.server.application.ApplicationStopped) {
        log.info("Stopping segmentation processor")
        segmentationProcessor.close()
    }
}