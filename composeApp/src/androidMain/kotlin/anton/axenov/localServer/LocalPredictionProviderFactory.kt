package anton.axenov.localServer

import android.content.Context
import anton.axenov.OnnxModelCatalog
import anton.axenov.SegmentationPredictionPipeline
import anton.axenov.SegmentationPredictionProvider
import anton.axenov.SegmentationPredictionTarget
import anton.axenov.SplitSegmentationPredictionProvider
import anton.axenov.createAndroidOnnxTargetPredictor
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp

/**
 * Builds one local prediction provider for the requested pipeline profile.
 *
 * @param context Android context used to load packaged ONNX assets.
 * @param pipeline selected predictor pipeline.
 * @param pythonBaseUrl Python fallback service base URL.
 * @return provider used by [anton.axenov.localServer.LocalClient].
 */
fun createLocalPredictionProvider(
    context: Context,
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
            zonePredictor = createAndroidOnnxTargetPredictor(
                context = context,
                modelSpec = OnnxModelCatalog.zonesYoloSeg,
            ),
            pointPredictor = createAndroidOnnxTargetPredictor(
                context = context,
                modelSpec = OnnxModelCatalog.pointsYoloSeg,
            ),
        )

        SegmentationPredictionPipeline.YOLO_SEG_FOR_ZONES_AND_STARDIST_POINTS -> SplitSegmentationPredictionProvider(
            zonePredictor = createAndroidOnnxTargetPredictor(
                context = context,
                modelSpec = OnnxModelCatalog.zonesYoloSeg,
            ),
            pointPredictor = createAndroidOnnxTargetPredictor(
                context = context,
                modelSpec = OnnxModelCatalog.pointsYoloStarDist,
            ),
        )
    }
}
