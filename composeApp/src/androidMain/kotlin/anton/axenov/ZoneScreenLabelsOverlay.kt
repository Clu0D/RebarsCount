package anton.axenov

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.absoluteOffset
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * Draws zone metric labels as screen-space text tracked to projected world points.
 *
 * @param labels projected label payloads.
 * @param modifier overlay modifier.
 */
@Composable
fun ZoneScreenLabelsOverlay(
    labels: List<ZoneScreenLabelEntry>,
    modifier: Modifier = Modifier,
) {
    Box(modifier = modifier.fillMaxSize()) {
        labels.forEach { label ->
            Text(
                text = label.text,
                color = Color.White,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .absoluteOffset {
                        IntOffset(
                            x = label.xPx.roundToInt() - LABEL_ANCHOR_OFFSET_X_PX,
                            y = label.yPx.roundToInt() - LABEL_ANCHOR_OFFSET_Y_PX,
                        )
                    }
                    .background(Color.Black.copy(alpha = 0.72f))
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            )
        }
    }
}

private const val LABEL_ANCHOR_OFFSET_X_PX = 90
private const val LABEL_ANCHOR_OFFSET_Y_PX = 24
