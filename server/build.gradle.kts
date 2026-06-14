import org.gradle.api.tasks.testing.Test

plugins {
    alias(libs.plugins.kotlinJvm)
    alias(libs.plugins.ktor)
    application
}

group = "anton.axenov"
version = "1.0.0"
application {
    mainClass.set("anton.axenov.ApplicationKt")
    
    val isDevelopment: Boolean = project.ext.has("development")
    applicationDefaultJvmArgs = listOf("-Dio.ktor.development=$isDevelopment")
}

dependencies {
    implementation(projects.shared)
    implementation(libs.logback)
    implementation(libs.ktor.clientContentNegotiation)
    implementation(libs.ktor.clientOkHttp)
    implementation(libs.ktor.serializationKotlinxJson)
    implementation(libs.ktor.serverCore)
    implementation(libs.ktor.serverContentNegotiation)
    implementation(libs.ktor.serverNetty)
    implementation(libs.ktor.serverSerializationKotlinxJson)
    testImplementation(libs.ktor.clientContentNegotiation)
    testImplementation(libs.ktor.clientMock)
    testImplementation(libs.ktor.serializationKotlinxJson)
    testImplementation(libs.ktor.serverTestHost)
    testImplementation(kotlin("test-junit5"))
    testImplementation(libs.kotest.runner.junit5)
    testImplementation(libs.kotest.assertions.core)
    testImplementation(libs.mockk)
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
