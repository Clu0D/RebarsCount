package anton.axenov

import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier

/**
 * Renders desktop fallback because ARCore is Android-only in this sample.
 *
 * @param modifier root layout modifier.
 * @param horizontalAlignment alignment for fallback textual content.
 */
@Composable
actual fun ArSceneHost(
    modifier: Modifier,
    horizontalAlignment: Alignment.Horizontal,
) {
    ArFallbackMessage(
        modifier = modifier,
        message = "AR sample is available on Android only.",
        horizontalAlignment = horizontalAlignment,
    )
}
