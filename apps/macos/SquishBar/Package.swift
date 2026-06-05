// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SquishBar",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "SquishBar",
            path: "Sources/SquishBar",
            resources: [
                .process("Assets.xcassets")
            ],
            linkerSettings: [
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
