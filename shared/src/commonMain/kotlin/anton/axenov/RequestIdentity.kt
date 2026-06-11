package anton.axenov

import kotlin.random.Random

/**
 * HTTP header carrying the stable client session identifier.
 */
const val SESSION_ID_HTTP_HEADER = "X-Session-Id"

/**
 * Generates a random identifier suitable for request and session correlation.
 *
 * @param random source of randomness.
 * @return hexadecimal identifier.
 */
fun generateRequestIdentifier(
    random: Random = Random.Default,
): String {
    return buildString {
        repeat(4) { index ->
            if (index > 0) {
                append('-')
            }
            append(random.nextLong().toULong().toString(16).padStart(16, '0'))
        }
    }
}
