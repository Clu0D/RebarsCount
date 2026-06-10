package anton.axenov

/**
 * Selects where point recognition and session result processing run.
 */
enum class ProcessingMode {
    FULLY_SERVER,
    FULLY_LOCAL,
    DEFERRED,
}

/**
 * Returns the user-visible Russian title for this processing mode.
 *
 * @return mode title.
 */
fun ProcessingMode.title(): String {
    return when (this) {
        ProcessingMode.FULLY_SERVER -> "Полностью серверная обработка"
        ProcessingMode.FULLY_LOCAL -> "Полностью локальная обработка"
        ProcessingMode.DEFERRED -> "Отложенная обработка"
    }
}
