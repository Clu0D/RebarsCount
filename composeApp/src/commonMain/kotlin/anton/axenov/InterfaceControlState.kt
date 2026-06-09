package anton.axenov

/**
 * User-selectable mode for correcting detected AR scene objects.
 */
enum class CorrectionMode {
    DELETE_ZONE,
    DELETE_POINT,
    ADD_POINT,
    MOVE_POINT_TO_ANOTHER_ZONE,
}

/**
 * Current state of placeholder controls described by the application workflow.
 *
 * @param isFrameSavingEnabled true when camera frames should be saved.
 * @param isZoneAdditionEnabled true when interest-zone addition should run.
 * @param isPointRecognitionEnabled true when rebar-end recognition should run.
 * @param isFullResultVisible true when the complete result view is selected.
 * @param correctionMode currently selected correction mode or null when correction is disabled.
 */
data class InterfaceControlState(
    val isFrameSavingEnabled: Boolean = false,
    val isZoneAdditionEnabled: Boolean = false,
    val isPointRecognitionEnabled: Boolean = false,
    val isFullResultVisible: Boolean = false,
    val correctionMode: CorrectionMode? = null,
) {
    /**
     * Toggles frame saving.
     *
     * @return state with the inverted frame-saving flag.
     */
    fun toggleFrameSaving(): InterfaceControlState {
        return copy(isFrameSavingEnabled = !isFrameSavingEnabled)
    }

    /**
     * Toggles interest-zone addition.
     *
     * @return state with the inverted zone-addition flag.
     */
    fun toggleZoneAddition(): InterfaceControlState {
        return copy(isZoneAdditionEnabled = !isZoneAdditionEnabled)
    }

    /**
     * Toggles rebar-end recognition.
     *
     * @return state with the inverted point-recognition flag.
     */
    fun togglePointRecognition(): InterfaceControlState {
        return copy(isPointRecognitionEnabled = !isPointRecognitionEnabled)
    }

    /**
     * Toggles between the AR scene and complete result view.
     *
     * @return state with the inverted complete-result visibility flag.
     */
    fun toggleFullResult(): InterfaceControlState {
        return copy(isFullResultVisible = !isFullResultVisible)
    }

    /**
     * Selects one correction mode or disables it when the selected mode is clicked again.
     *
     * @param mode correction mode clicked by the user.
     * @return state with [mode] selected exclusively or null when it was already selected.
     */
    fun toggleCorrectionMode(mode: CorrectionMode): InterfaceControlState {
        return copy(correctionMode = mode.takeUnless { it == correctionMode })
    }
}
