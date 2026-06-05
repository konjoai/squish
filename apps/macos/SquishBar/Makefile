# SquishBar Makefile — builds a self-contained macOS .app bundle and DMG
# Requires: Xcode 15+ (swift 5.9+, macOS 13+ SDK), hdiutil (built-in)

SCHEME    = SquishBar
APPNAME   = SquishBar.app
DMG_NAME  = SquishBar.dmg
BUNDLE_ID = com.konjoai.SquishBar
BUILD_DIR = .build
VOL_NAME  = SquishBar

.PHONY: all clean run release dmg

all: $(APPNAME)

ASSETS    = Sources/SquishBar/Assets.xcassets

$(APPNAME):
	swift build -c release 2>&1
	@mkdir -p "$(APPNAME)/Contents/MacOS"
	@mkdir -p "$(APPNAME)/Contents/Resources"
	@cp "$(BUILD_DIR)/release/$(SCHEME)" "$(APPNAME)/Contents/MacOS/$(SCHEME)"
	@cp "Resources/Info.plist" "$(APPNAME)/Contents/Info.plist"
	@# Compile asset catalog → Assets.car + AppIcon.icns into the .app's Resources/
	@xcrun actool "$(ASSETS)" \
		--compile "$(APPNAME)/Contents/Resources" \
		--platform macosx \
		--minimum-deployment-target 13.0 \
		--app-icon AppIcon \
		--output-partial-info-plist "$(BUILD_DIR)/assetcatalog_generated_info.plist" \
		--output-format human-readable-text \
		> /dev/null
	@echo "  Compiled Assets.car"
	@echo "✓ Built $(APPNAME)"
	@echo "  Open with:  open $(APPNAME)"

release: $(APPNAME)
	@echo "✓ Release build complete: $(APPNAME)"

# Create a distributable DMG containing the .app
dmg: $(APPNAME)
	@echo "▸ Creating $(DMG_NAME)…"
	@rm -f "$(DMG_NAME)"
	hdiutil create -volname "$(VOL_NAME)" \
		-srcfolder "$(APPNAME)" \
		-ov -format UDZO \
		"$(DMG_NAME)"
	@echo "✓ $(DMG_NAME) ready  ($(shell du -sh $(DMG_NAME) | cut -f1))"

run: $(APPNAME)
	open $(APPNAME)

clean:
	rm -rf $(APPNAME) $(DMG_NAME) $(BUILD_DIR)
