if not exist .\venv\Scripts\activate.bat (
  python3.exe -m venv venv
)

call .\venv\Scripts\activate.bat

pip.exe install -r requirements.txt

pause