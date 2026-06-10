import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import org.gradle.api.GradleException
import org.gradle.api.tasks.testing.Test
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.util.Properties

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
    alias(libs.plugins.composeHotReload)
}

kotlin {
    androidTarget {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }
    
    listOf(
        iosArm64(),
        iosSimulatorArm64()
    ).forEach { iosTarget ->
        iosTarget.binaries.framework {
            baseName = "ComposeApp"
            isStatic = true
        }
    }
    
    jvm()
    
    sourceSets {
        androidMain.dependencies {
            implementation(libs.compose.uiToolingPreview)
            implementation(libs.androidx.activity.compose)
            implementation(libs.ktor.clientOkHttp)
            implementation(libs.google.arcore)
            implementation(libs.opencv)
            implementation(libs.sceneview.arsceneview)
            implementation(libs.commons.math3)
            implementation(libs.jts.core)
        }
        androidUnitTest.dependencies {
            implementation(libs.kotest.runner.junit5)
            implementation(libs.kotest.assertions.core)
            implementation(libs.mockk)
        }
        commonMain.dependencies {
            implementation(libs.compose.runtime)
            implementation(libs.compose.foundation)
            implementation(libs.compose.material3)
            implementation(libs.compose.ui)
            implementation(libs.compose.components.resources)
            implementation(libs.compose.uiToolingPreview)
            implementation(libs.androidx.lifecycle.viewmodelCompose)
            implementation(libs.androidx.lifecycle.runtimeCompose)
            implementation(libs.korlibs.math)
            implementation(libs.ktor.clientCore)
            implementation(libs.ktor.clientContentNegotiation)
            implementation(libs.ktor.serializationKotlinxJson)
            implementation(libs.kotlinx.serialization.json)
            implementation(projects.shared)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotest.assertions.core)
        }
        jvmMain.dependencies {
            implementation(compose.desktop.currentOs)
            implementation(libs.kotlinx.coroutinesSwing)
            implementation(libs.commons.math3)
            implementation(libs.jts.core)
        }
        jvmTest.dependencies {
            implementation(libs.kotest.runner.junit5)
            implementation(libs.kotest.assertions.core)
            implementation(libs.ktor.clientMock)
            implementation(libs.mockk)
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}

val filamentSourceAssetsDir = layout.projectDirectory.dir("src/androidMain/assets/materials")
val filamentGeneratedAssetsDir = layout.buildDirectory.dir("generated/filament/materials")
val defaultRepoMatcPath = rootProject.layout.projectDirectory.file(".tools/filament/1.68.2-host/filament/bin/matc").asFile.path
val localProperties = Properties().apply {
    val localPropertiesFile = rootProject.file("local.properties")
    if (localPropertiesFile.exists()) {
        localPropertiesFile.inputStream().use(::load)
    }
}
val filamentMatcPathProvider = providers.gradleProperty("filament.matc.path")
    .orElse(providers.provider { localProperties.getProperty("filament.matc.path") })
    .orElse(providers.environmentVariable("FILAMENT_MATC"))
    .orElse(providers.provider { defaultRepoMatcPath })

val compileFilamentMaterials = tasks.register<CompileFilamentMaterialsTask>("compileFilamentMaterials") {
    group = "build"
    description = "Compiles Filament .mat sources into runtime .filamat assets."
    sourceDir.set(filamentSourceAssetsDir)
    outputDir.set(filamentGeneratedAssetsDir)
    configuredMatcPath.convention(filamentMatcPathProvider.orElse(""))
}

android {
    namespace = "anton.axenov"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "anton.axenov"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.targetSdk.get().toInt()
        versionCode = 1
        versionName = "1.0"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    sourceSets["main"].assets.srcDir(filamentGeneratedAssetsDir)
}

tasks.named("preBuild") {
    dependsOn(compileFilamentMaterials)
}

dependencies {
    debugImplementation(libs.compose.uiTooling)
}

compose.desktop {
    application {
        mainClass = "anton.axenov.MainKt"

        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "anton.axenov"
            packageVersion = "1.0.0"
        }
    }
}
