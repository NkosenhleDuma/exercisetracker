@echo off
REM Remove src directory from PYTHONPATH when conda environment is deactivated (Windows)

REM Get the project root directory (parent of .exercisetracker)
REM Script is at .exercisetracker\etc\conda\deactivate.d\unset_path.bat
REM So we need to go up 4 levels: deactivate.d -> conda -> etc -> .exercisetracker -> project root
for %%I in ("%~dp0..\..\..\..") do set "PROJECT_ROOT=%%~fI"
set "SRC_DIR=%PROJECT_ROOT%\src"

REM Remove src from PYTHONPATH
if defined PYTHONPATH (
    set "PYTHONPATH=%PYTHONPATH:%SRC_DIR%;=%"
    set "PYTHONPATH=%PYTHONPATH:;%SRC_DIR%=%"
    set "PYTHONPATH=%PYTHONPATH:%SRC_DIR%=%"
    echo Removed %SRC_DIR% from PYTHONPATH
)

