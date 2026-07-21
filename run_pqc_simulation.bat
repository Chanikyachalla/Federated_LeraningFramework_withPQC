@echo off
title PQC Security Simulation
set BASE=%~dp0
set PATH=%BASE%mingw_portable\mingw64\bin;%BASE%cmake_portable\cmake-3.30.8-windows-x86_64\bin;%USERPROFILE%\_oqs\bin;%PATH%
echo.
echo  Starting PQC Security Simulation...
echo.
py -c "import os,sys; os.environ['PATH']=r'%USERPROFILE%\_oqs\bin' + ';' + r'%BASE%mingw_portable\mingw64\bin' + ';' + os.environ.get('PATH',''); os.chdir(r'%BASE%'); sys.path.insert(0,r'%BASE%'); exec(open('pqc_simulation.py').read())"
pause
