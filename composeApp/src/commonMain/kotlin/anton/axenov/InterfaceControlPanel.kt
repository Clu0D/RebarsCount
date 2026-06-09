package anton.axenov

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import artest.composeapp.generated.resources.Res
import artest.composeapp.generated.resources.control_add_point
import artest.composeapp.generated.resources.control_delete_point
import artest.composeapp.generated.resources.control_delete_zone
import artest.composeapp.generated.resources.control_frame_saving
import artest.composeapp.generated.resources.control_full_result
import artest.composeapp.generated.resources.control_move_point
import artest.composeapp.generated.resources.control_point_recognition
import artest.composeapp.generated.resources.control_zone_addition
import org.jetbrains.compose.resources.DrawableResource
import org.jetbrains.compose.resources.painterResource

/**
 * Draws placeholder workflow and mutually exclusive correction controls.
 *
 * @param state current interface control state.
 * @param onStateChanged callback invoked with state produced by a button click.
 * @param modifier panel modifier.
 */
@Composable
fun InterfaceControlPanel(
    state: InterfaceControlState,
    onStateChanged: (InterfaceControlState) -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.72f))
            .horizontalScroll(rememberScrollState())
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        ControlButton(
            icon = Res.drawable.control_frame_saving,
            description = if (state.isFrameSavingEnabled) {
                "Остановить сохранение кадров"
            } else {
                "Запустить сохранение кадров"
            },
            isSelected = state.isFrameSavingEnabled,
            onClick = { onStateChanged(state.toggleFrameSaving()) },
        )
        ControlButton(
            icon = Res.drawable.control_zone_addition,
            description = if (state.isZoneAdditionEnabled) {
                "Остановить добавление зон"
            } else {
                "Запустить добавление зон"
            },
            isSelected = state.isZoneAdditionEnabled,
            onClick = { onStateChanged(state.toggleZoneAddition()) },
        )
        ControlButton(
            icon = Res.drawable.control_point_recognition,
            description = if (state.isPointRecognitionEnabled) {
                "Остановить распознавание торцов"
            } else {
                "Запустить распознавание торцов"
            },
            isSelected = state.isPointRecognitionEnabled,
            onClick = { onStateChanged(state.togglePointRecognition()) },
        )
        ControlButton(
            icon = Res.drawable.control_full_result,
            description = if (state.isFullResultVisible) "Вернуться к AR-сцене" else "Просмотреть полный результат",
            isSelected = state.isFullResultVisible,
            onClick = { onStateChanged(state.toggleFullResult()) },
        )
        CorrectionMode.entries.forEach { mode ->
            ControlButton(
                icon = mode.icon(),
                description = mode.description(),
                isSelected = state.correctionMode == mode,
                onClick = { onStateChanged(state.toggleCorrectionMode(mode)) },
            )
        }
    }
}

/**
 * Draws one selected or inactive control button.
 *
 * @param icon square PNG icon resource.
 * @param description accessibility description of the represented action.
 * @param isSelected true when the represented mode is active.
 * @param onClick callback invoked by a click.
 */
@Composable
private fun ControlButton(
    icon: DrawableResource,
    description: String,
    isSelected: Boolean,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        modifier = Modifier.size(CONTROL_BUTTON_SIZE),
        contentPadding = PaddingValues(CONTROL_BUTTON_PADDING),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isSelected) {
                MaterialTheme.colorScheme.primary
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            },
            contentColor = if (isSelected) {
                MaterialTheme.colorScheme.onPrimary
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
        ),
    ) {
        Image(
            painter = painterResource(icon),
            contentDescription = description,
            modifier = Modifier.size(CONTROL_ICON_SIZE),
        )
    }
}

/**
 * Returns the icon resource for a correction mode.
 *
 * @return square PNG icon resource.
 */
private fun CorrectionMode.icon(): DrawableResource {
    return when (this) {
        CorrectionMode.DELETE_ZONE -> Res.drawable.control_delete_zone
        CorrectionMode.DELETE_POINT -> Res.drawable.control_delete_point
        CorrectionMode.ADD_POINT -> Res.drawable.control_add_point
        CorrectionMode.MOVE_POINT_TO_ANOTHER_ZONE -> Res.drawable.control_move_point
    }
}

/**
 * Returns the accessibility description for a correction mode.
 *
 * @return Russian correction-mode description.
 */
private fun CorrectionMode.description(): String {
    return when (this) {
        CorrectionMode.DELETE_ZONE -> "Удалить зону"
        CorrectionMode.DELETE_POINT -> "Удалить ошибочный торец"
        CorrectionMode.ADD_POINT -> "Добавить торец вручную"
        CorrectionMode.MOVE_POINT_TO_ANOTHER_ZONE -> "Перенести торец в другую связку"
    }
}

private val CONTROL_BUTTON_SIZE = 64.dp
private val CONTROL_ICON_SIZE = 44.dp
private val CONTROL_BUTTON_PADDING = 8.dp
