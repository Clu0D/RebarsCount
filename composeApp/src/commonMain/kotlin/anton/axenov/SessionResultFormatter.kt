package anton.axenov

/**
 * Builds user-visible information about the current processing session.
 *
 * @param zones zones currently stored in the session.
 * @return text containing the number of reconstructed points assigned to each zone.
 */
fun buildSessionResultText(zones: List<Zone>): String {
    if (zones.isEmpty()) {
        return "Результат сессии\n\nЗоны не обнаружены."
    }

    return buildString {
        appendLine("Результат сессии")
        appendLine()
        zones
            .sortedBy { zone -> zone.id }
            .forEach { zone ->
                appendLine("Зона ${zone.id}: ${zone.sceneWorldPointsCount} точек")
            }
    }.trimEnd()
}
