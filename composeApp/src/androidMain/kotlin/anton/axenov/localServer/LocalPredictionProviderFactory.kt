package anton.axenov.localServer

import anton.axenov.SegmentationPredictionPipeline
import anton.axenov.SegmentationPredictionProvider
import anton.axenov.SegmentationPredictionTarget
import anton.axenov.SplitSegmentationPredictionProvider
import anton.axenov.StubYoloSegTargetPredictor
import anton.axenov.StubYoloStarDistTargetPredictor
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp

/**
 * Builds one local prediction provider for the requested pipeline profile.
 *
 * @param pipeline selected predictor pipeline.
 * @param pythonBaseUrl Python fallback service base URL.
 * @return provider used by [anton.axenov.localServer.LocalClient].
 */
fun createLocalPredictionProvider(
    pipeline: SegmentationPredictionPipeline,
    pythonBaseUrl: String,
): SegmentationPredictionProvider {
    return when (pipeline) {
        SegmentationPredictionPipeline.PYTHON -> SplitSegmentationPredictionProvider(
            zonePredictor = PythonSegmentationPredictor(
                baseUrl = pythonBaseUrl,
                target = SegmentationPredictionTarget.ZONES,
                httpClient = HttpClient(OkHttp),
            ),
            pointPredictor = PythonSegmentationPredictor(
                baseUrl = pythonBaseUrl,
                target = SegmentationPredictionTarget.POINTS,
                httpClient = HttpClient(OkHttp),
            ),
        )

        SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_POINTS -> SplitSegmentationPredictionProvider(
            zonePredictor = StubYoloSegTargetPredictor(SegmentationPredictionTarget.ZONES),
            pointPredictor = StubYoloSegTargetPredictor(SegmentationPredictionTarget.POINTS),
        )

        SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS -> SplitSegmentationPredictionProvider(
            zonePredictor = StubYoloSegTargetPredictor(SegmentationPredictionTarget.ZONES),
            pointPredictor = StubYoloStarDistTargetPredictor(SegmentationPredictionTarget.POINTS),
        )
    }
}
