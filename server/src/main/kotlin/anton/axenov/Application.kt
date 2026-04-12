package anton.axenov

import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlinx.serialization.json.Json

private val snapshotsByZoneId = ConcurrentHashMap<Long, CopyOnWriteArrayList<ZoneSnapshotUploadDto>>()

/**
 * Clears in-memory server state.
 */
fun resetServerState() {
    snapshotsByZoneId.clear()
}

fun main() {
    embeddedServer(Netty, port = SERVER_PORT, host = "0.0.0.0", module = Application::module)
        .start(wait = true)
}

/**
 * Configures HTTP routes for the AR snapshot server.
 */
fun Application.module() {
    install(ContentNegotiation) {
        json(
            Json {
                ignoreUnknownKeys = true
                prettyPrint = true
            },
        )
    }

    routing {
        get("/") {
            call.respond(ServerHealthResponseDto(ok = true, message = "Ktor server is online"))
        }

        get("/health") {
            call.respond(ServerHealthResponseDto(ok = true, message = "Ktor server is online"))
        }

        post("/snapshots") {
            val payload = call.receive<ZoneSnapshotUploadDto>()
            val zoneSnapshots = snapshotsByZoneId.getOrPut(payload.zone.id) { CopyOnWriteArrayList() }
            zoneSnapshots += payload
            call.respond(
                SnapshotUploadResponseDto(
                    ok = true,
                    zoneId = payload.zone.id,
                    snapshotCount = zoneSnapshots.size,
                    message = "stored snapshot for zone ${payload.zone.id}",
                ),
            )
        }

        get("/zone-statuses") {
            val knownZoneIds = snapshotsByZoneId.keys.toSortedSet()
            val statuses = knownZoneIds.map { zoneId ->
                ZoneStatusDto(
                    zone = zoneId,
                    text = "${snapshotsByZoneId[zoneId]?.size ?: 0} snapshot(s) uploaded",
                )
            }
            call.respond(statuses)
        }
    }
}