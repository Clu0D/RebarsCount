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
    implementation(libs.ktor.serverCore)
    implementation(libs.ktor.serverContentNegotiation)
    implementation(libs.ktor.serverNetty)
    implementation(libs.ktor.serverSerializationKotlinxJson)
    testImplementation(libs.ktor.clientContentNegotiation)
    testImplementation(libs.ktor.serializationKotlinxJson)
    testImplementation(libs.ktor.serverTestHost)
    testImplementation(libs.kotlin.testJunit)
}