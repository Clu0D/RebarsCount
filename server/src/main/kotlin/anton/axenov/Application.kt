package anton.axenov

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.KotlinxSerializationConverter
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json

private var activePredictor: SegmentationPredictionProvider = createServerPredictionProvider()
private var segmentationSessions = SegmentationSessionRegistry(
    predictor = activePredictor,
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
        segmentationSessions.clear()
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
    predictor: SegmentationPredictionProvider = createServerPredictionProvider(),
) {
    runBlocking {
        segmentationSessions.close()
    }
    activePredictor.close()
    activePredictor = predictor
    segmentationSessions = SegmentationSessionRegistry(
        predictor = activePredictor,
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
            val sessionId = call.sessionIdOrRespond() ?: return@post
            val payload = call.receive<ZoneSnapshotUploadDto>()
            if (payload.sessionId != sessionId) {
                call.respond(
                    HttpStatusCode.BadRequest,
                    ServerHealthResponse(
                        ok = false,
                        message = "Payload sessionId does not match request session header",
                    ),
                )
                return@post
            }
            call.respond(segmentationSessions.processorFor(sessionId).predictPoints(payload))
        }

        post("/predict_zones") {
            val sessionId = call.sessionIdOrRespond() ?: return@post
            val payload = call.receive<DetectionFrameSnapshotDto>()
            val prediction = try {
                segmentationSessions.processorFor(sessionId).predictZones(payload)
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
            val sessionId = call.sessionIdOrRespond() ?: return@get
            call.respond(segmentationSessions.processorFor(sessionId).fetchZoneStatuses())
        }

        get("/world-points") {
            val sessionId = call.sessionIdOrRespond() ?: return@get
            call.respond(segmentationSessions.processorFor(sessionId).fetchWorldPoints())
        }

        post("/world-points/add") {
            val sessionId = call.sessionIdOrRespond() ?: return@post
            val payload = call.receive<AddWorldPointDto>()
            call.respond(segmentationSessions.processorFor(sessionId).addWorldPoint(payload))
        }

        post("/world-points/delete") {
            val sessionId = call.sessionIdOrRespond() ?: return@post
            val payload = call.receive<WorldPointIdDto>()
            call.respond(segmentationSessions.processorFor(sessionId).deleteWorldPoint(payload.pointId))
        }

        post("/world-points/rotate-zone") {
            val sessionId = call.sessionIdOrRespond() ?: return@post
            val payload = call.receive<WorldPointIdDto>()
            call.respond(segmentationSessions.processorFor(sessionId).rotateWorldPointZone(payload.pointId))
        }

        post("/zones/delete") {
            val sessionId = call.sessionIdOrRespond() ?: return@post
            val payload = call.receive<ZoneIdDto>()
            call.respond(segmentationSessions.processorFor(sessionId).deleteZone(payload.zoneId))
        }

        post("/delete_request") {
            val sessionId = call.sessionIdOrRespond() ?: return@post
            val payload = call.receive<DeleteRequestDto>()
            call.respond(segmentationSessions.processorFor(sessionId).deleteRequest(payload.requestId))
        }

        post("/start_new_session") {
            val sessionId = call.sessionIdOrRespond() ?: return@post
            call.respond(segmentationSessions.processorFor(sessionId).startNewSession())
        }
    }

    monitor.subscribe(io.ktor.server.application.ApplicationStopped) {
        log.info("Stopping segmentation processor")
        runBlocking {
            segmentationSessions.close()
        }
        activePredictor.close()
    }
}

/**
 * Returns current session identifier from the request or sends `400 Bad Request`.
 *
 * @return non-empty session identifier or null when the request is invalid.
 */
private suspend fun ApplicationCall.sessionIdOrRespond(): String? {
    val sessionId = request.headers[SESSION_ID_HTTP_HEADER]
        ?.takeIf { headerValue -> headerValue.isNotBlank() }
        ?: run {
            respond(
                HttpStatusCode.BadRequest,
                ServerHealthResponse(
                    ok = false,
                    message = "Missing $SESSION_ID_HTTP_HEADER header",
                ),
            )
            return null
        }
    return sessionId
}

/**
 * Stores one isolated processor per client session identifier.
 *
 * @param predictor shared prediction adapter used by all session processors.
 * @param onSnapshotAccepted optional debug callback invoked for every accepted snapshot.
 */
private class SegmentationSessionRegistry(
    private val predictor: SegmentationPredictionProvider,
    private val onSnapshotAccepted: (filename: String, payload: ZoneSnapshotUploadDto) -> Unit,
) {
    private val stateMutex = Mutex()
    private val processorsBySessionId = mutableMapOf<String, SegmentationSessionProcessor>()

    /**
     * Returns processor state for one client session, creating it on first access.
     *
     * @param sessionId stable client session identifier.
     * @return processor dedicated to this session.
     */
    suspend fun processorFor(sessionId: String): SegmentationSessionProcessor {
        return stateMutex.withLock {
            processorsBySessionId.getOrPut(sessionId) {
                SegmentationSessionProcessor(
                    predictor = predictor,
                    sessionId = sessionId,
                    onSnapshotAccepted = onSnapshotAccepted,
                    closePredictorOnClose = false,
                )
            }
        }
    }

    /**
     * Clears all known session processors and their queues.
     */
    suspend fun clear() {
        stateMutex.withLock {
            val processors = processorsBySessionId.values.toList()
            processorsBySessionId.clear()
            processors.forEach { processor -> processor.close() }
        }
    }

    /**
     * Closes all session processors.
     */
    suspend fun close() {
        clear()
    }
}
