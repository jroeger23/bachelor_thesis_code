@echo off
if not exist .\venv\Scripts\activate.bat (
  echo "No venv found. Run setup_venv.bat first."
  pause
  exit 1
)

call .\venv\Scripts\activate.bat

cmd.exe