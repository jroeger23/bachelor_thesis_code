#!/bin/sh

activate_script=".venv/bin/activate"

if [ -e "$activate_script" ]; then
  :
else
  echo "No .venv found. Creating new venv"
  python3 -m venv .venv
fi

source "$activate_script"

pip install -e .
pip install -r requirements.txt