package anton.axenov


/**
 * Detects zones of interest on a screenshot.
 *
 * @param segmentationServerClient client used to request server-side zone prediction.
 */
class DetectInterestZones(
    private val segmentationServerClient: SegmentationClient,
) {
    /**
     * Detects zones of interest on a screenshot.
     *
     * @param snapshot snapshot captured from camera frame.
     * @return list of detected zone bounding boxes.
     */
    suspend fun detectZones(snapshot: DetectionFrameSnapshot): List<DetectedInterestZone> {
        val width = snapshot.imageWidth
        val height = snapshot.imageHeight
        if (width <= 1 || height <= 1) {
            return emptyList()
        }

        val response = segmentationServerClient.predictZones(snapshot.toPayload())
        return response.instances.mapNotNull { instance ->
            val box = instance.bbox
            val clampedBox = ScreenBoundingBox(
                left = box.x.coerceIn(0, width - 1),
                top = box.y.coerceIn(0, height - 1),
                right = (box.x + box.width).coerceIn(0, width - 1),
                bottom = (box.y + box.height).coerceIn(0, height - 1),
            )
            if (clampedBox.left >= clampedBox.right || clampedBox.top >= clampedBox.bottom) {
                null
            } else {
                DetectedInterestZone(
                    screenBoundingBox = clampedBox,
                    confidence = instance.confidence
                )
            }
        }
    }
}
