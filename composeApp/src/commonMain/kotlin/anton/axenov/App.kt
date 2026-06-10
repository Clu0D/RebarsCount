package anton.axenov

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

/**
 * Renders the main application surface.
 */
@Composable
@Preview
fun App() {
    var selectedModeName by rememberSaveable { mutableStateOf<String?>(null) }
    val selectedMode = selectedModeName?.let(ProcessingMode::valueOf)

    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background,
        ) {
            if (selectedMode == null) {
                ProcessingModeSelector(
                    onModeSelected = { mode -> selectedModeName = mode.name },
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                ArSceneHost(
                    processingMode = selectedMode,
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                )
            }
        }
    }
}

/**
 * Displays the initial choice between supported processing modes.
 *
 * @param onModeSelected callback invoked with the selected mode.
 * @param modifier root layout modifier.
 */
@Composable
fun ProcessingModeSelector(
    onModeSelected: (ProcessingMode) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Выберите режим обработки",
            style = MaterialTheme.typography.headlineSmall,
        )
        ProcessingMode.entries.forEach { mode ->
            Button(onClick = { onModeSelected(mode) }) {
                Text(mode.title())
            }
        }
    }
}
