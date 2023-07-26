#include <stdio.h>
#include"../inc/MandelSet.cu.h"

#define LERP(a,b,t) (a + t * (b - a))  //linear interpolation
#define FRACT(x) ((x) - floor(x))


//CUDA KERNEL DEFINITION
//global kernel function declaration
__global__ void Mandelbrot_kernel(unsigned char* pos, unsigned int maxWidth, unsigned int maxHeight, float zoom, float xCenter, float yCenter, int maxIterations, float4 cu_outerColor1, float4 cu_outerColor2)
{
	//variable declarations
	float4 innerColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	float4 outerColor1 = cu_outerColor1;
	float4 outerColor2 = cu_outerColor2;
	float4 color;

	int offset;
	float real;
	float imag;
	float cReal;
	float cImag;
	float dist;
	int iter;
	float tmp_real;
	float tmp_imag;

	/*unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;*/

	unsigned int x = blockIdx.x;
	unsigned int y = blockIdx.y;

	if (x >= maxWidth)
		return;

	if (y >= maxHeight)
		return;

	offset = x + y * gridDim.x;

	float fx = (float)x / (float)maxWidth;
	float fy = (float)y / (float)maxHeight;

	real = ((float)maxWidth / (float)maxHeight) * fx * zoom + xCenter;
	imag = fy * zoom + yCenter;

	cReal = real;
	cImag = imag;

	for (iter = 0; (iter < maxIterations); iter++)
	{
		tmp_real = (real * real) - (imag * imag);
		tmp_imag = (2.0 * real * imag);

		real = tmp_real + cReal;
		imag = tmp_imag + cImag;

		dist = (real * real) + (imag * imag);

		if (dist > 16.0)
			break;
	}

	if (dist < 4.0)
		color = innerColor;
	else
	{
		color.x = LERP(outerColor1.x, outerColor2.x, FRACT(iter * 0.013f));
		color.y = LERP(outerColor1.y, outerColor2.y, FRACT(iter * 0.013f));
		color.z = LERP(outerColor1.z, outerColor2.z, FRACT(iter * 0.013f));
		color.w = LERP(outerColor1.w, outerColor2.w, FRACT(iter * 0.013f));
		//color = LERP(outerColor1, outerColor2, FRACT(iter * 0.015f));
	}

	pos[offset * 4 + 0] = 255 * color.x;
	pos[offset * 4 + 1] = 255 * color.y;
	pos[offset * 4 + 2] = 255 * color.z;
	pos[offset * 4 + 3] = 255 * color.w;
	
}

void launchCUDAKernel(unsigned char* pos, unsigned int maxWidth, unsigned int maxHeight, float zoom, float xCenter, float yCenter, int maxIterations, float4 cu_outerColor1, float4 cu_outerColor2)
{
	dim3 grid(maxWidth, maxHeight);
	//dim3 block(16, 16);
	Mandelbrot_kernel << <grid, 1 >> > (pos, maxWidth, maxHeight, zoom, xCenter, yCenter, maxIterations, cu_outerColor1, cu_outerColor2);

}

