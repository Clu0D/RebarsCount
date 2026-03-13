package anton.axenov

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.exceptions.UnavailableException
import io.github.sceneview.ar.ARScene
import kotlinx.coroutines.cancel

/**
 * Renders Android AR sample content.
 *
 * @param modifier root layout modifier.
 * @param horizontalAlignment alignment for fallback textual content.
 */
@Composable
actual fun ArSceneHost(
    modifier: Modifier,
    horizontalAlignment: Alignment.Horizontal,
) {
    val context = LocalContext.current
    val activity = context.findActivity()
    val lifecycleOwner = LocalLifecycleOwner.current
    var hasCameraPermission by remember { mutableStateOf(context.hasCameraPermission()) }
    var requestedOnce by rememberSaveable { mutableStateOf(false) }
    var arCoreInstallRequested by rememberSaveable { mutableStateOf(false) }
    var arSetupStatus by rememberSaveable { mutableStateOf("AR setup: checking") }
    var isArReady by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    DisposableEffect(lifecycleOwner, context, activity, arCoreInstallRequested, hasCameraPermission) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                hasCameraPermission = context.hasCameraPermission()
                if (hasCameraPermission) {
                    refreshAndApplyArCoreSetup(
                        context = context,
                        activity = activity,
                        requestInstallIfNeeded = !arCoreInstallRequested,
                        onReadyChanged = { isArReady = it },
                        onStatusChanged = { arSetupStatus = it },
                        onInstallRequested = { arCoreInstallRequested = true },
                    )
                }
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    LaunchedEffect(hasCameraPermission, arCoreInstallRequested, context, activity) {
        if (!hasCameraPermission && !requestedOnce) {
            requestedOnce = true
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
        hasCameraPermission = context.hasCameraPermission()
        if (hasCameraPermission) {
            refreshAndApplyArCoreSetup(
                context = context,
                activity = activity,
                requestInstallIfNeeded = !arCoreInstallRequested,
                onReadyChanged = { isArReady = it },
                onStatusChanged = { arSetupStatus = it },
                onInstallRequested = { arCoreInstallRequested = true },
            )
        }
    }

    if (!hasCameraPermission) {
        PermissionRequestContent(
            modifier = modifier,
            horizontalAlignment = horizontalAlignment,
            isPermanentlyDenied = requestedOnce && (activity?.isCameraPermissionPermanentlyDenied() ?: false),
            onRequestPermission = { permissionLauncher.launch(Manifest.permission.CAMERA) },
            onOpenSettings = { context.openApplicationSettings() },
        )
        return
    }

    if (!isArReady) {
        ArSetupContent(
            modifier = modifier,
            horizontalAlignment = horizontalAlignment,
            statusText = arSetupStatus,
            onRetry = {
                refreshAndApplyArCoreSetup(
                    context = context,
                    activity = activity,
                    requestInstallIfNeeded = true,
                    onReadyChanged = { isArReady = it },
                    onStatusChanged = { arSetupStatus = it },
                    onInstallRequested = { arCoreInstallRequested = true },
                )
            },
        )
        return
    }

    var debugText by remember { mutableStateOf("AR: waiting for renderer") }
    val coroutineScope = rememberCoroutineScope()
    val statusReporter = remember {
        SceneStatusReporter { debugMessage ->
            debugText = debugMessage
        }
    }
    val detectionPipeline = remember(coroutineScope, statusReporter) {
        ArDetectionPipeline(
            coroutineScope = coroutineScope,
            reportStatus = { message, force -> statusReporter.report(message, force) },
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            detectionPipeline.onSceneDisposed()
            coroutineScope.cancel()
        }
    }

    Box(modifier = modifier) {
        ARScene(
            modifier = Modifier.fillMaxSize(),
            planeRenderer = false,
            sessionConfiguration = { session, config ->
                if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    config.depthMode = Config.DepthMode.AUTOMATIC
                }
            },
            onViewCreated = {
                detectionPipeline.onSceneCreated(this)
                statusReporter.report("AR view created (SceneView)")
            },
            onSessionCreated = {
                statusReporter.report("Session created")
            },
            onSessionResumed = {
                statusReporter.report("Session resumed")
            },
            onSessionPaused = {
                statusReporter.report("Session paused")
            },
            onSessionFailed = { exception ->
                statusReporter.report("Session failed: ${exception.javaClass.simpleName}")
            },
            onTrackingFailureChanged = { reason: TrackingFailureReason? ->
                if (reason != null && reason != TrackingFailureReason.NONE) {
                    statusReporter.report("Camera tracking failure: ${reason.name}", force = true)
                }
            },
            onSessionUpdated = { _, frame ->
                detectionPipeline.onSessionUpdated(frame)
            },
        )
        Text(
            text = debugText,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(12.dp)
                .background(Color.Black.copy(alpha = 0.7f))
                .padding(8.dp),
            color = Color.White,
        )
    }
}

/**
 * Runs ARCore setup checks and optionally requests ARCore installation.
 *
 * @param context Android context.
 * @param activity host activity.
 * @param requestInstallIfNeeded true when install dialog can be shown.
 * @return setup result with readiness and user-visible status.
 */
private fun refreshArCoreSetup(
    context: Context,
    activity: Activity?,
    requestInstallIfNeeded: Boolean,
): ArCoreSetupResult {
    val result = ensureArCoreReady(
        context = context,
        activity = activity,
        requestInstallIfNeeded = requestInstallIfNeeded,
    )
    val message = if (result.isReady) {
        "Using ARCore backend"
    } else {
        result.message
    }
    return result.copy(message = message)
}

/**
 * Executes ARCore setup refresh and applies returned state through provided callbacks.
 *
 * @param context Android context.
 * @param activity host activity.
 * @param requestInstallIfNeeded true when install dialog can be shown.
 * @param onReadyChanged callback that receives setup readiness.
 * @param onStatusChanged callback that receives user-visible setup status.
 * @param onInstallRequested callback invoked when install flow was requested.
 */
private fun refreshAndApplyArCoreSetup(
    context: Context,
    activity: Activity?,
    requestInstallIfNeeded: Boolean,
    onReadyChanged: (Boolean) -> Unit,
    onStatusChanged: (String) -> Unit,
    onInstallRequested: () -> Unit,
) {
    val setupResult = refreshArCoreSetup(
        context = context,
        activity = activity,
        requestInstallIfNeeded = requestInstallIfNeeded,
    )
    applyArCoreSetupResult(
        setupResult = setupResult,
        onReadyChanged = onReadyChanged,
        onStatusChanged = onStatusChanged,
        onInstallRequested = onInstallRequested,
    )
}

/**
 * Applies one ARCore setup result to UI state callbacks.
 *
 * @param setupResult setup result to apply.
 * @param onReadyChanged callback that receives setup readiness.
 * @param onStatusChanged callback that receives user-visible setup status.
 * @param onInstallRequested callback invoked when install flow was requested.
 */
private fun applyArCoreSetupResult(
    setupResult: ArCoreSetupResult,
    onReadyChanged: (Boolean) -> Unit,
    onStatusChanged: (String) -> Unit,
    onInstallRequested: () -> Unit,
) {
    onReadyChanged(setupResult.isReady)
    onStatusChanged(setupResult.message)
    if (setupResult.installRequested) {
        onInstallRequested()
    }
}

/**
 * Shows permission controls and retry actions when camera access is missing.
 *
 * @param modifier root layout modifier.
 * @param horizontalAlignment horizontal alignment for content.
 * @param isPermanentlyDenied true when the system no longer shows permission dialogs.
 * @param onRequestPermission callback to launch permission request.
 * @param onOpenSettings callback to open app settings.
 */
@Composable
private fun PermissionRequestContent(
    modifier: Modifier,
    horizontalAlignment: Alignment.Horizontal,
    isPermanentlyDenied: Boolean,
    onRequestPermission: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    CenteredContentColumn(
        modifier = modifier.fillMaxSize(),
        horizontalAlignment = horizontalAlignment,
    ) {
        Text("Camera permission is required for AR rendering.")
        Button(onClick = onRequestPermission) {
            Text("Grant Camera Permission")
        }
        if (isPermanentlyDenied) {
            Text("Permission was denied with \"Don't ask again\".")
            Button(onClick = onOpenSettings) {
                Text("Open App Settings")
            }
        }
    }
}

/**
 * Finds the nearest hosting activity from a context chain.
 *
 * @return hosting activity or null.
 */
private fun Context.findActivity(): Activity? {
    var current: Context? = this
    while (current is ContextWrapper) {
        if (current is Activity) {
            return current
        }
        current = current.baseContext
    }
    return null
}

/**
 * Checks whether camera permission is granted.
 *
 * @return true when granted.
 */
private fun Context.hasCameraPermission(): Boolean {
    return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED
}

/**
 * Detects when camera permission is denied permanently by the user.
 *
 * @return true when system dialog will not be shown anymore.
 */
private fun Activity.isCameraPermissionPermanentlyDenied(): Boolean {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED
    ) {
        return false
    }
    return !ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)
}

/**
 * Opens application settings page where permission can be enabled manually.
 */
private fun Context.openApplicationSettings() {
    val intent = Intent(
        Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
        Uri.fromParts("package", packageName, null),
    ).apply {
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
    startActivity(intent)
}

/**
 * Shows AR setup status and retry action.
 *
 * @param modifier root layout modifier.
 * @param horizontalAlignment horizontal alignment for content.
 * @param statusText current setup status.
 * @param onRetry callback to retry setup checks.
 */
@Composable
private fun ArSetupContent(
    modifier: Modifier,
    horizontalAlignment: Alignment.Horizontal,
    statusText: String,
    onRetry: () -> Unit,
) {
    CenteredContentColumn(
        modifier = modifier.fillMaxSize(),
        horizontalAlignment = horizontalAlignment,
    ) {
        Text("AR setup is required.")
        Text(statusText)
        Button(onClick = onRetry) {
            Text("Retry AR Setup")
        }
    }
}

/**
 * Renders a vertically centered column used by permission/setup fallback screens.
 *
 * @param modifier root layout modifier.
 * @param horizontalAlignment horizontal alignment for children.
 * @param content composable children rendered inside centered column.
 */
@Composable
private fun CenteredContentColumn(
    modifier: Modifier,
    horizontalAlignment: Alignment.Horizontal,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = modifier.fillMaxSize(),
        horizontalAlignment = horizontalAlignment,
        verticalArrangement = Arrangement.Center,
        content = content,
    )
}

/**
 * Tries to make ARCore ready by checking support and requesting installation when needed.
 *
 * @param context Android context.
 * @param activity host activity.
 * @param requestInstallIfNeeded true when install dialog may be shown.
 * @return setup result that can be displayed to user.
 */
private fun ensureArCoreReady(
    context: Context,
    activity: Activity?,
    requestInstallIfNeeded: Boolean,
): ArCoreSetupResult {
    if (activity == null) {
        return ArCoreSetupResult(
            isReady = false,
            message = "ARCore setup failed: activity is null",
        )
    }

    val availability = ArCoreApk.getInstance().checkAvailability(context)
    if (availability.isTransient) {
        return ArCoreSetupResult(
            isReady = false,
            message = "ARCore availability check is in progress",
        )
    }
    if (!availability.isSupported) {
        return ArCoreSetupResult(
            isReady = false,
            message = "This device does not support ARCore ($availability)",
        )
    }

    return try {
        when (ArCoreApk.getInstance().requestInstall(activity, requestInstallIfNeeded)) {
            ArCoreApk.InstallStatus.INSTALLED -> ArCoreSetupResult(
                isReady = true,
                message = "ARCore installed",
            )

            ArCoreApk.InstallStatus.INSTALL_REQUESTED -> ArCoreSetupResult(
                isReady = false,
                installRequested = true,
                message = "ARCore install/update requested. Complete it and return.",
            )
        }
    } catch (error: UnavailableException) {
        ArCoreSetupResult(
            isReady = false,
            message = "ARCore unavailable: ${error.javaClass.simpleName}",
        )
    } catch (error: Exception) {
        ArCoreSetupResult(
            isReady = false,
            message = "ARCore setup error: ${error.javaClass.simpleName}",
        )
    }
}

/**
 * Delivers SceneView debug lines to UI on the main thread with light deduplication.
 *
 * @param onStatusChanged consumer used by compose overlay.
 */
private class SceneStatusReporter(
    private val onStatusChanged: (String) -> Unit,
) {
    private val mainHandler = Handler(Looper.getMainLooper())
    private var lastMessage: String = ""
    private var lastReportAtMs: Long = 0

    /**
     * Reports a status message.
     *
     * @param message message text.
     * @param force true to bypass deduplication and cooldown.
     */
    fun report(message: String, force: Boolean = false) {
        val now = SystemClock.elapsedRealtime()
        if (!force && message == lastMessage && now - lastReportAtMs < STATUS_REPEAT_COOLDOWN_MS) {
            return
        }
        lastMessage = message
        lastReportAtMs = now
        mainHandler.post {
            onStatusChanged(message)
        }
    }
}

/**
 * Result of ARCore setup step.
 *
 * @param isReady true when ARCore can be used.
 * @param installRequested true when install flow was triggered.
 * @param message user-visible status.
 */
private data class ArCoreSetupResult(
    val isReady: Boolean,
    val installRequested: Boolean = false,
    val message: String,
)

private const val STATUS_REPEAT_COOLDOWN_MS = 1000L
