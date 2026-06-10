package anton.axenov

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/**
 * Displays current session information above the entire AR interface.
 *
 * @param resultText formatted current-session information.
 * @param deferredProcessingState current deferred-processing state or null outside deferred mode.
 * @param onProcessSavedFrames callback invoked to start processing saved frames.
 * @param onClose callback invoked by the close button.
 * @param modifier overlay modifier.
 */
@Composable
fun FullResultOverlay(
    resultText: String,
    deferredProcessingState: DeferredProcessingState? = null,
    onProcessSavedFrames: () -> Unit = {},
    onClose: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .clickable(onClick = {}),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(FULL_RESULT_CONTENT_PADDING),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = resultText,
                color = MaterialTheme.colorScheme.onBackground,
                style = MaterialTheme.typography.headlineSmall,
            )
            if (deferredProcessingState != null) {
                Button(
                    onClick = onProcessSavedFrames,
                    enabled = deferredProcessingState.canStart,
                ) {
                    Text("Обработать сохранённые ${deferredProcessingState.savedFramesCount} кадров")
                }
                Text(
                    text = deferredProcessingState.statusText,
                    color = MaterialTheme.colorScheme.onBackground,
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }
        Button(
            onClick = onClose,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(FULL_RESULT_CLOSE_PADDING),
        ) {
            Text(
                text = "×",
                color = Color.White,
                style = MaterialTheme.typography.headlineSmall,
            )
        }
    }
}

private val FULL_RESULT_CONTENT_PADDING = 24.dp
private val FULL_RESULT_CLOSE_PADDING = 12.dp

/**
 * Current progress of deferred server processing.
 *
 * @param savedFramesCount number of currently stored frame-zone payloads.
 * @param processedFramesCount number of payloads completed or failed on the server.
 * @param isRunning true while deferred processing is active.
 * @param errorMessage latest deferred-processing error or null.
 */
data class DeferredProcessingState(
    val savedFramesCount: Int = 0,
    val processedFramesCount: Int = 0,
    val isRunning: Boolean = false,
    val errorMessage: String? = null,
) {
    val canStart: Boolean
        get() = savedFramesCount > 0 && !isRunning

    val progressPercent: Int
        get() = if (savedFramesCount == 0) {
            0
        } else {
            (processedFramesCount.coerceIn(0, savedFramesCount) * 100) / savedFramesCount
        }

    val statusText: String
        get() = when {
            errorMessage != null -> "Ошибка: $errorMessage"
            isRunning -> "Обработано фотографий: $progressPercent% ($processedFramesCount/$savedFramesCount)"
            processedFramesCount > 0 -> "Обработано фотографий: $progressPercent% ($processedFramesCount/$savedFramesCount)"
            else -> "Сохранено фотографий: $savedFramesCount"
        }
}
