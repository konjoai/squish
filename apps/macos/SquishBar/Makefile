# SquishBar Makefile — builds a self-contained macOS .app bundle
# Requires: Xcode 15+ (swift 5.9+, macOS 13+ SDK)

SCHEME    = SquishBar
APPNAME   = SquishBar.app
BUNDLE_ID = com.wesleyscholl.SquishBar
BUILD_DIR = .build

.PHONY: all clean run

all: $(APPNAME)

$(APPNAME):
	swift build -c release 2>&1
	@mkdir -p "$(APPNAME)/Contents/MacOS"
	@mkdir -p "$(APPNAME)/Contents/Resources"
	@cp "$(BUILD_DIR)/release/$(SCHEME)" "$(APPNAME)/Contents/MacOS/$(SCHEME)"
	@cp "Resources/Info.plist" "$(APPNAME)/Contents/Info.plist"
	@echo "✓ Built $(APPNAME)"
	@echo "  Open with:  open $(APPNAME)"

run: $(APPNAME)
	open $(APPNAME)

clean:
	rm -rf $(APPNAME) $(BUILD_DIR)
