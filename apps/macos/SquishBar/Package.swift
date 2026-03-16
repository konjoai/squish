// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SquishBar",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "SquishBar",
            path: "Sources/SquishBar",
            linkerSettings: [
                // Embed Info.plist so macOS treats the binary as a UI app
                // (required for MenuBarExtra / NSStatusItem to work at runtime)
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Resources/Info.plist",
                ]),
            ]
        ),
    ]
)
