package anton.axenov

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
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
 * @param onClose callback invoked by the close button.
 * @param modifier overlay modifier.
 */
@Composable
fun FullResultOverlay(
    resultText: String,
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
        ) {
            Text(
                text = resultText,
                color = MaterialTheme.colorScheme.onBackground,
                style = MaterialTheme.typography.headlineSmall,
            )
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
