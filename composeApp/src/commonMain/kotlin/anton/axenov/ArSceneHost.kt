package anton.axenov

import androidx.compose.foundation.layout.Box
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier

/**
 * Hosts the platform-specific AR content.
 *
 * @param processingMode selected processing mode.
 * @param modifier root layout modifier.
 * @param horizontalAlignment alignment for fallback textual content.
 */
@Composable
expect fun ArSceneHost(
    processingMode: ProcessingMode,
    modifier: Modifier = Modifier,
    horizontalAlignment: Alignment.Horizontal = Alignment.CenterHorizontally,
)

/**
 * Draws a fallback message for platforms without the Android AR implementation.
 *
 * @param modifier root layout modifier.
 * @param message fallback text.
 * @param horizontalAlignment horizontal alignment for text.
 */
@Composable
fun ArFallbackMessage(
    modifier: Modifier = Modifier,
    message: String,
    horizontalAlignment: Alignment.Horizontal = Alignment.CenterHorizontally,
) {
    Box(
        modifier = modifier,
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = message,
            modifier = Modifier.align(Alignment.Center),
        )
    }
}
