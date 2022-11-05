#!/bin/sh

activate_script=".venv/bin/activate"

if [ -e "$activate_script" ]; then
  source "$activate_script"
else
  echo "No .venv found. Run setup_venv.sh to create one."
  exit 1
fi