#!/usr/bin/env bash
# Run once from the project root:  bash scripts/setup_linux.sh
set -e
sudo apt update
sudo apt install -y python3-venv python3-pip python3-tk git libportaudio2 \
  libsndfile1 xdg-utils tesseract-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo
echo "Done. Next time, activate the environment with:  source .venv/bin/activate"
echo "Launch the GUI with:  python main.py"
