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
     * @return list of detected zone polygons.
     */
    suspend fun detectZones(snapshot: DetectionFrameSnapshot): List<DetectedInterestZone> {
        val width = snapshot.imageWidth
        val height = snapshot.imageHeight
        if (width <= 1 || height <= 1) {
            return emptyList()
        }

        val response = segmentationServerClient.predictZones(snapshot.toPayload())
        return response.instances.mapNotNull { instance ->
            val clampedPolygon = instance.polygon
                .map { point ->
                    ImagePoint(
                        x = point.x.coerceIn(0, width - 1),
                        y = point.y.coerceIn(0, height - 1),
                    )
                }
                .distinct()
            if (clampedPolygon.size < 3) {
                null
            } else {
                DetectedInterestZone(
                    screenPolygon = clampedPolygon,
                    confidence = instance.confidence
                )
            }
        }
    }
}
