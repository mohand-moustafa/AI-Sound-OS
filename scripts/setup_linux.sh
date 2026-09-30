#!/usr/bin/env bash
# Run once from the project root:  bash scripts/setup_linux.sh
set -e
sudo apt update
sudo apt install -y python3-venv python3-pip git libportaudio2 xdg-utils
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo
echo "Done. Next time, activate the environment with:  source .venv/bin/activate"
echo "Test it with:  python main.py --text --dry-run"
