cls

nvcc -c -o MandelSet.cu.obj src\MandelSet.cu

cl.exe /c /EHsc /I "C:\glew\include" /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include" /I "E:\SHRUTI\AstroMediComp\AssetsAndTools\Libraries\freetype-2.11.0\include" /I "C:\SOIL\inc" src\MandelSetCUDAInterop.cpp

rc.exe res\Resources.rc

link.exe /OUT:"MandelSetCUDAInterop.exe" *.obj res\Resources.res user32.lib gdi32.lib /LIBPATH:"C:\glew\lib\Release\x64" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64" /LIBPATH:"E:\SHRUTI\AstroMediComp\AssetsAndTools\Libraries\freetype-2.11.0\build\Debug" /LIBPATH:"C:\SOIL\lib\Debug" /MACHINE:X64 /SUBSYSTEM:WINDOWS  msvcrtd.lib libcmt.lib

DEL *.obj
DEL res\Resources.res

MandelSetCUDAInterop.exe

