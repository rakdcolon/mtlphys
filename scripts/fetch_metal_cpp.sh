#!/usr/bin/env bash
# Download Apple's metal-cpp headers into vendor/metal-cpp/.
# Apple sometimes rotates the archive name with each macOS release; if the
# pinned URL stops working, grab the latest from https://developer.apple.com/metal/cpp/
# and unzip its contents into vendor/metal-cpp/ such that
# vendor/metal-cpp/Metal/Metal.hpp exists.
set -euo pipefail

URL="https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/vendor/metal-cpp"
TMP="$(mktemp -d)"

if [[ -f "$TARGET/Metal/Metal.hpp" ]]; then
    echo "metal-cpp already installed at $TARGET — nothing to do."
    exit 0
fi

echo "Downloading metal-cpp from $URL"
curl -fsSL -o "$TMP/metal-cpp.zip" "$URL"

echo "Unzipping into $TARGET"
mkdir -p "$ROOT/vendor"
unzip -q "$TMP/metal-cpp.zip" -d "$ROOT/vendor"
rm -rf "$TMP"

if [[ ! -f "$TARGET/Metal/Metal.hpp" ]]; then
    echo "Error: expected $TARGET/Metal/Metal.hpp after unzip — archive layout may have changed."
    echo "Check https://developer.apple.com/metal/cpp/ and unpack manually."
    exit 1
fi

echo "Done. Now run:  xcodegen generate"
