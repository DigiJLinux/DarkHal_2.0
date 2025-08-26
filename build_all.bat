@echo off
echo Building DarkHal 2.0 for all platforms...
echo (Windows x64, Linux x64, Linux ARM64)
echo.
python build_all_platforms.py all
pause