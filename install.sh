#!/usr/bin/env bash
set -e

# Resolve the directory this script lives in (even if called via symlink)
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP_NAME="NumViewer"
APP_ID="numviewer"
DESKTOP_FILE="${APP_ID}.desktop"
DESKTOP_DIR="$HOME/.local/share/applications"

EXEC="$APP_DIR/numviewer.sh"
ICON="$APP_DIR/icon.png"

# Make sure the app is executable
chmod +x "$EXEC"

mkdir -p "$DESKTOP_DIR"

cat > "$DESKTOP_DIR/$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$APP_NAME
Comment=A lightweight tool for visualizing NumPy arrays
Exec="$EXEC" %f
Icon=$ICON
Terminal=false
Categories=Science;Utility;
MimeType=inode/directory;
StartupNotify=true
EOF

chmod +x "$DESKTOP_DIR/$DESKTOP_FILE"

update-desktop-database "$DESKTOP_DIR" || true

echo "Installed $APP_NAME launcher."
echo "You can now find it in your application menu."
