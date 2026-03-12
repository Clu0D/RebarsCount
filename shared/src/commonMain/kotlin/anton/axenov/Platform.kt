package anton.axenov

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform