#!/usr/bin/env bash
# Remove dictation autostart and launcher (keeps build artifacts and models).
set -euo pipefail

echo "==> Stopping dictation"
systemctl --user stop whisper-dictation.service 2>/dev/null || true
systemctl --user disable whisper-dictation.service 2>/dev/null || true
rm -f "${HOME}/.config/systemd/user/whisper-dictation.service"
systemctl --user daemon-reload 2>/dev/null || true

pkill -f "scripts/dictation/dictation.py" 2>/dev/null || true

rm -f "${HOME}/.local/bin/whisper-dictation"
rm -f "${HOME}/.config/autostart/whisper-dictation.desktop"

echo "Removed autostart and launcher."
echo "Config kept at ~/.config/whisper-dictation/"
echo "Repo build/ and models/ were not deleted."
