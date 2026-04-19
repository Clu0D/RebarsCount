import org.gradle.api.DefaultTask
import org.gradle.api.GradleException
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputDirectory
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.process.ExecOperations
import java.io.File
import javax.inject.Inject

/**
 * Resolves a matc path that may be configured in Windows form inside local.properties.
 *
 * When Gradle runs under Windows the original path is used as-is. When Gradle runs under WSL,
 * a path like D:\repo\tool\matc.exe is translated into /mnt/d/repo/tool/matc.exe.
 *
 * @param configuredPath raw configured path from Gradle properties, local.properties or env.
 * @return executable path for the current host OS.
 */
fun resolveMatcPath(configuredPath: String): String {
    val trimmedPath = configuredPath.trim()
    val isWindowsHost = System.getProperty("os.name").startsWith("Windows", ignoreCase = true)
    if (isWindowsHost) {
        return trimmedPath
    }

    val windowsDriveMatch = Regex("""^([A-Za-z]):\\(.*)$""").matchEntire(trimmedPath)
        ?: return trimmedPath
    val driveLetter = windowsDriveMatch.groupValues[1].lowercase()
    val windowsTail = windowsDriveMatch.groupValues[2].replace('\\', '/')
    return "/mnt/$driveLetter/$windowsTail"
}

/**
 * Converts a Windows absolute path to a WSL path when needed.
 *
 * @param path filesystem path in current Gradle process notation.
 * @return WSL-compatible path.
 */
fun toWslPath(path: String): String {
    val normalizedPath = path.replace('\\', '/')
    val windowsDriveMatch = Regex("""^([A-Za-z]):/(.*)$""").matchEntire(normalizedPath)
        ?: return normalizedPath
    val driveLetter = windowsDriveMatch.groupValues[1].lowercase()
    val windowsTail = windowsDriveMatch.groupValues[2]
    return "/mnt/$driveLetter/$windowsTail"
}

/**
 * Compiles Filament material source files into runtime .filamat assets.
 */
abstract class CompileFilamentMaterialsTask : DefaultTask() {

    /**
     * Exec service used to invoke the material compiler without touching Task.project at execution time.
     */
    @get:Inject
    abstract val execOperations: ExecOperations

    /**
     * Directory containing source .mat files.
     */
    @get:InputDirectory
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sourceDir: DirectoryProperty

    /**
     * Directory where generated .filamat files are written.
     */
    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    /**
     * Path to the matc executable.
     */
    @get:Input
    abstract val configuredMatcPath: Property<String>

    /**
     * Compiles all source material files into Filament runtime packages.
     */
    @TaskAction
    fun compileMaterials() {
        val sourceRoot = sourceDir.get().asFile
        val sourceFiles = sourceDir.asFileTree.matching { include("**/*.mat") }.files.sorted()
        if (sourceFiles.isEmpty()) {
            return
        }

        val configuredPath = configuredMatcPath.get().trim()
        if (configuredPath.isEmpty()) {
            throw GradleException(
                "Filament material sources were found, but matc is not configured. " +
                        "Set filament.matc.path in local.properties, FILAMENT_MATC, " +
                        "or -Pfilament.matc.path to the matc executable path."
            )
        }
        val resolvedMatcPath = resolveMatcPath(configuredPath)
        if (!File(resolvedMatcPath).exists()) {
            throw GradleException(
                "Configured matc executable does not exist: $configuredPath " +
                        "(resolved for current host as $resolvedMatcPath)."
            )
        }

        val generatedOutputDir = outputDir.get().asFile.apply { mkdirs() }
        val isWindowsHost = System.getProperty("os.name").startsWith("Windows", ignoreCase = true)

        sourceFiles.forEach { sourceFile ->
            val relativePath = sourceFile.relativeTo(sourceRoot).invariantSeparatorsPath
            val outputFile = File(
                generatedOutputDir,
                relativePath.removeSuffix(".mat") + ".filamat",
            )
            outputFile.parentFile.mkdirs()

            execOperations.exec {
                if (isWindowsHost && !configuredPath.endsWith(".exe", ignoreCase = true)) {
                    commandLine(
                        "wsl.exe",
                        toWslPath(resolvedMatcPath),
                        "-p", "mobile",
                        "-a", "all",
                        "-o", toWslPath(outputFile.absolutePath),
                        toWslPath(sourceFile.absolutePath),
                    )
                } else {
                    commandLine(
                        resolvedMatcPath,
                        "-p", "mobile",
                        "-a", "all",
                        "-o", outputFile.absolutePath,
                        sourceFile.absolutePath,
                    )
                }
            }
        }
    }
}
