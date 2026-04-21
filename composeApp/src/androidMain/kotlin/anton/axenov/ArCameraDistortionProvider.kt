package anton.axenov

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import com.google.ar.core.Session
import java.util.concurrent.ConcurrentHashMap

/**
 * Gets distortion coefficients for the active ARCore camera.
 *
 * @param context Android context used to obtain [CameraManager].
 */
class ArCameraDistortionProvider(
    context: Context,
) {
    private val cameraManager = context.applicationContext.getSystemService(CameraManager::class.java)
    private val distortionByCameraId = ConcurrentHashMap<String, List<Float>>()

    /**
     * Returns distortion coefficients for the current ARCore camera.
     *
     * @param session active ARCore session.
     * @return distortion coefficients or empty list when unavailable.
     */
    fun distortionCoefficients(session: Session): List<Float> {
        val cameraId = runCatching { session.cameraConfig.cameraId }.getOrNull() ?: return emptyList()
        return distortionByCameraId.getOrPut(cameraId) {
            loadDistortionCoefficients(cameraId)
        }
    }

    /**
     * Finds distortion coefficients for the current ARCore camera.
     *
     * @param cameraId active camera identifier returned by ARCore.
     * @return distortion coefficients or empty list when unavailable.
     */
    private fun loadDistortionCoefficients(cameraId: String): List<Float> {
        val characteristics = runCatching {
            cameraManager.getCameraCharacteristics(cameraId)
        }.getOrNull() ?: return emptyList()

        val distortion = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            characteristics.get(CameraCharacteristics.LENS_DISTORTION)
        } else {
            // CameraCharacteristics.LENS_RADIAL_DISTORTION does not match opencv
            null
        }
        return distortion?.toList().orEmpty()
    }
}
