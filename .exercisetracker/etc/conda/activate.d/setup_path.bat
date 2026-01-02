@echo off
REM Add src directory to PYTHONPATH when conda environment is activated (Windows)

REM Get the project root directory (parent of .exercisetracker)
REM Script is at .exercisetracker\etc\conda\activate.d\setup_path.bat
REM So we need to go up 4 levels: activate.d -> conda -> etc -> .exercisetracker -> project root
for %%I in ("%~dp0..\..\..\..") do set "PROJECT_ROOT=%%~fI"
set "SRC_DIR=%PROJECT_ROOT%\src"

REM Add src to PYTHONPATH if not already present
echo %PYTHONPATH% | findstr /C:"%SRC_DIR%" >nul
if errorlevel 1 (
    if defined PYTHONPATH (
        set "PYTHONPATH=%SRC_DIR%;%PYTHONPATH%"
    ) else (
        set "PYTHONPATH=%SRC_DIR%"
    )
    echo Added %SRC_DIR% to PYTHONPATH
)

