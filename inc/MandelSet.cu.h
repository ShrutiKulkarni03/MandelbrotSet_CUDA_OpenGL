#pragma once

void launchCUDAKernel(unsigned char* pos, unsigned int maxWidth, unsigned int maxHeight, float zoom, float xCenter, float yCenter, int maxIterations, float4 cu_outerColor1, float4 cu_outerColor2);

//float outerColorArray[4];

