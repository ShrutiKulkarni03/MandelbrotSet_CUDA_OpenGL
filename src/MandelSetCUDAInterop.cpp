//header files
#include<windows.h>
#include<stdio.h>

#include<gl\glew.h>
#include<gl\wglew.h>
#include<GL/GL.h>

#include <SOIL.h>

#include "../inc/Resources.h"
#include "../inc/vmath.h"

//freetype
#include "ft2build.h"
#include FT_FREETYPE_H
#include<map>
#include<string>

//cuda
#include<cuda_gl_interop.h>
#include<cuda_runtime.h>
#include "../inc/MandelSet.cu.h"

//macro functions LIBS
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "freetyped.lib")
#pragma comment(lib, "soil.lib")
#pragma comment(lib, "Winmm.lib")

#define WIN_WIDTH 1280
#define WIN_HEIGHT 720

#define MAX_WIDTH 1920
#define MAX_HEIGHT 1080

#define MAX_TEXT_ALPHA_ELEMENTS 50

#define LERP(a,b,t) (a + t * (b - a))  //linear interpolation
#define FRACT(x) ((x) - floor(x))

namespace vmath
{
	vmath::vec4 operator*(float x, vmath::vec4 v)
	{
		vmath::vec4 ret;
		ret[0] = x * v[0];
		ret[1] = x * v[1];
		ret[2] = x * v[2];
		ret[3] = x * v[3];

		return ret;
	}
};

using namespace vmath;

//PROPERTIES OF VERTEX:
enum
{
	SPK_ATTRIBUTE_POSITION = 0,
	SPK_ATTRIBUTE_COLOR,
	SPK_ATTRIBUTE_NORMAL,
	SPK_ATTRIBUTE_TEXCOORD,
};

//text
struct Character {
	unsigned int textTextureID;     //ID handle of texture
	vec2 size;						//size of glyph
	vec2 bearing;					//offset from baseline to left/top of glyph
	unsigned int advance;			//horizontal offset to advance to next glyph
};

std::map<GLchar, Character> characters;

struct TextAlphaValue 
{
	float alpha[MAX_TEXT_ALPHA_ELEMENTS];

}textAlphaVal;


//global function declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void Initialize(void);
GLuint LoadImageAsTexture(const char*);
void Display(void);
void Update(void);
void Uninitialize(void);
void launchCPUKernel(vec4, vec4);

//global variable declaration
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };
bool gbFullscreen = false;
HWND ghwnd = NULL;
bool gbActiveWindow = false;
FILE* gpFile = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;
bool PlaySong = false;

//interop
cudaError_t cudaResult;
struct cudaGraphicsResource* cuda_graphics_resource_pos = NULL;
struct cudaGraphicsResource* cuda_graphics_resource_tex = NULL;
unsigned int maxWidth = 1920;
unsigned int maxHeight = 1080;
bool bOnGPU = false;
unsigned char* gpu_tex;
float4 cuOuterColor1 = make_float4(0.0f, 0.11f, 0.22f, 1.0f);
float4 cuOuterColor2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
vec4 cpuOuterColor1 = vec4(0.0f, 0.11f, 0.22f, 1.0f);
vec4 cpuOuterColor2 = vec4(1.0f, 1.0f, 1.0f, 1.0f);

//fps
LARGE_INTEGER start_counter, end_counter, counts, frequency, fps, ms;
char fps_str[128] = "fps: 0";
char ms_str[128] = "ms: 0";
char iter_str[128] = "iterations: 0";

float currentTime = 0.0f;

//mandelbrot passthrough shader
GLuint shaderProgramObject;
GLuint mvpUniform;
GLuint texSamplerUniform;

//fullscreen quad shader
GLuint quadShaderProgramObject;
GLuint color_alpha_uniform;
GLuint sampler_tex_uniform;

//text shader
GLuint textShaderProgramObject;
GLuint textSamplerUniform;
GLuint textColorUniform;
GLuint mvpMatrixUniformText;

//VAOs and VBOs
//mandelbrot vao
GLuint vao;                          //vertex array object
GLuint vbo_pos_quad;
GLuint vbo_tex_quad;
GLuint vbo_col_quad;

//text vao
GLuint vao_text;
GLuint vbo_text;

mat4 perspectiveProjectionMatrix;   //4x4 matrix

//texture variables
GLuint textureID;
GLuint soil_textureID;
GLuint texImage;
GLuint mandelTextureGPU;
GLuint mandelTextureCPU;

GLuint tex1_amc;
GLuint tex2_grp;
GLuint tex3_mandelbrot;
GLuint tex4_sk;
GLuint tex5_sir;
GLuint tex6_tech;
GLuint tex7_ref;
GLuint tex8_splTy;
GLuint tex9_ty;

int currentTexture = 1;

//for mandelbrot interop
GLubyte pos[MAX_WIDTH * MAX_HEIGHT * 4];

int maxIterations = 0;
float zoom = 3.0f;
float xOffset = 0.0f;
float yOffset = 0.0f;
float xCenter = -3.0f;
float yCenter = -1.5f;

//keys
bool iKeyPressed = false;
int sKeyPressed = 3;
float dt = 0.0f;
float quadAlphaVal = 0.0f;
int spaceKey_scene = 1;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//prototype
	void ToggleFullscreen(void);

	//local variable declaration
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MyApp");
	bool bDone = false;
	int x, y, width, height;

	if (fopen_s(&gpFile, "RenderLog.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Cannot Open The Desired File\n"), TEXT("Error"), MB_OK);
		exit(0);
	}
	else
	{
		fprintf(gpFile, ("Log File Created Successfully, Program Started Successfully.\n\n"));
	}

	//code
	width = GetSystemMetrics(SM_CXSCREEN);
	height = GetSystemMetrics(SM_CYSCREEN);

	x = (width / 2) - (WIN_WIDTH / 2);
	y = (height / 2) - (WIN_HEIGHT / 2);

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));

	RegisterClassEx(&wndclass);

	//CreateWindow
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("My Application - SHRUTI KULKARNI"),
		(WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE),
		x,
		y,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	//funtion call
	Initialize();
	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	//for default fullscreen
	if (gbFullscreen == false)
		ToggleFullscreen();

	//FPS count
	int fpsCount = 0;

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start_counter);

	//game loop
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				//update function for OpenGL rendering
				Update();
				//display function for OpenGL rendering
				Display();

			}
		}

		fpsCount++;

		QueryPerformanceCounter(&end_counter);

		counts.QuadPart = end_counter.QuadPart - start_counter.QuadPart;

		fps.QuadPart = frequency.QuadPart / counts.QuadPart;
		ms.QuadPart = ((1000 * counts.QuadPart) / frequency.QuadPart);

		if (ms.QuadPart > 1000)
		{
			start_counter = end_counter;

			sprintf(fps_str, "fps: %d", fpsCount);
			sprintf(ms_str, "seconds: %f", (float)1.0 / fpsCount);
		
			fpsCount = 0;
		}

		//fprintf(gpFile, "fps: %lld, ms: %lld\n", fps.QuadPart, ms.QuadPart);
		//fprintf(gpFile, "fps: %d, seconds: %f\n", fpsCount, (float)1.0 / fpsCount);
		//SetWindowTextA(hwnd, fps_str);

	}
	return(msg.wParam);
}

//WndProc
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//function declaration
	void ToggleFullscreen(void);
	void Resize(int, int);

	char str[128];

	switch (iMsg)
	{
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_ERASEBKGND:
		return(0);

	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		//xCenter
		case VK_RIGHT:				
			xCenter += 0.01f;
			break;
		case VK_LEFT:				
			xCenter -= 0.01f;
			break;

		//yCenter
		case VK_UP:				
			yCenter += 0.01f;
			break;
		case VK_DOWN:			
			yCenter -= 0.01f;
			break;

		//next scene
		case VK_SPACE:			
			spaceKey_scene++;
			if (spaceKey_scene >= 3)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 4;
				spaceKey_scene = 3;
			}
			break;

		//colors change

			//default blue + white
		case VK_NUMPAD0:			
			cuOuterColor1 = make_float4(0.0f, 0.11f, 0.22f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.11f, 0.22f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 1.0f, 1.0f, 1.0f);
			break;
		
			//black + red
		case VK_NUMPAD1:			
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 0.0f, 0.0f, 1.0f);
			break;

			//black + green
		case VK_NUMPAD2:
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(0.0f, 1.0f, 0.0f, 1.0f);
			break;

			//black + blue
		case VK_NUMPAD3:
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(0.0f, 0.0f, 1.0f, 1.0f);
			break;

			//black + cyan
		case VK_NUMPAD4:
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(0.0f, 1.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(0.0f, 1.0f, 1.0f, 1.0f);
			break;

			//black + magenta
		case VK_NUMPAD5:
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 0.0f, 1.0f, 1.0f);
			break;

			//black + yellow
		case VK_NUMPAD6:
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 1.0f, 0.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 1.0f, 0.0f, 1.0f);
			break;

			//black + orange
		case VK_NUMPAD7:
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 0.5f, 0.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 0.5f, 0.0f, 1.0f);
			break;


			//red + white
		case 49:
			cuOuterColor1 = make_float4(0.3f, 0.0f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.3f, 0.0f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 1.0f, 1.0f, 1.0f);
			break;

			//green + white
		case 50:
			cuOuterColor1 = make_float4(0.0f, 0.2f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.2f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 1.0f, 1.0f, 1.0f);
			break;

			//blue + white
		case 51:
			cuOuterColor1 = make_float4(0.0f, 0.0f, 0.3f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.0f, 0.0f, 0.3f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 1.0f, 1.0f, 1.0f);
			break;

			//magenta + white
		case 52:
			cuOuterColor1 = make_float4(0.3f, 0.0f, 0.3f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.3f, 0.0f, 0.3f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 1.0f, 1.0f, 1.0f);
			break;

			//orange + white
		case 53:
			cuOuterColor1 = make_float4(0.58f, 0.19f, 0.0f, 1.0f);
			cuOuterColor2 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			cpuOuterColor1 = vec4(0.58f, 0.19f, 0.0f, 1.0f);
			cpuOuterColor2 = vec4(1.0f, 1.0f, 1.0f, 1.0f);
			break;

		default:
			break;

		}
		break;

	case WM_CHAR:
		switch (wParam)
		{
		//code

		case 'F':
		case 'f':
			ToggleFullscreen();
			break;

		//cpu toggle
		case 'C':
		case 'c':
			bOnGPU = false;
			break;

		//gpu toggle
		case 'G':
		case 'g':
			bOnGPU = true;
			break;

		//iterations
		case 'I':
			iKeyPressed = true;
			maxIterations -= 10;
			break;
		case 'i':
			iKeyPressed = true;
			maxIterations += 10;
			break;

		//zoom animation
		case 'S':
			sKeyPressed = 0;
			break;
		case 's':
			sKeyPressed = 1;
			break;

		//reset
		case 'R':
		case 'r':
			zoom = 3.0f;
			xCenter = -3.0f;
			yCenter = -1.5f;
			dt = 0.0f;
			sKeyPressed = 0;
			break;

		//zoom manually
		case 'Z':
			zoom += 0.01f;
			break;
		case 'z':
			zoom -= 0.01f;
			break;

		default:
			iKeyPressed = false;
			sKeyPressed = 3;
			break;
		}
		/*sprintf(str, "%f, %f\n", iteration, mandelScale);*/
		sprintf(iter_str, "iterations: %d", maxIterations);
		sprintf(str, "zoom: %f, x: %f, y: %f", zoom, xCenter, yCenter);
		SetWindowTextA(hwnd, str);
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		Uninitialize();
		PostQuitMessage(0);
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));

}

void ToggleFullscreen(void)
{
	//local variable declaration
	MONITORINFO mi = { sizeof(MONITORINFO) };

	//code
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, (dwStyle & ~WS_OVERLAPPEDWINDOW));
				SetWindowPos(ghwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}

		}
		ShowCursor(false);
		gbFullscreen = true;
	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, (dwStyle | WS_OVERLAPPEDWINDOW));
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(true);
		gbFullscreen = false;
	}
}

void initFreetype(void)
{
	FT_Library ft;
	if (FT_Init_FreeType(&ft))
	{
		fprintf(gpFile, "Could not init FT Lib\n");
		fflush(gpFile);
	}

	std::string font_name = "res/fonts/Roboto-Black.ttf";
	if (font_name.empty())
	{
		fprintf(gpFile, "Could not load font_name\n");
		fflush(gpFile);
	}

	FT_Face face;
	if (FT_New_Face(ft, font_name.c_str(), 0, &face))
	{
		fprintf(gpFile, "Could not load font\n");
		fflush(gpFile);
	}
	else
	{
		FT_Set_Pixel_Sizes(face, 0, 48);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		for (unsigned char c = 0; c < 128; c++)
		{
			if (FT_Load_Char(face, c, FT_LOAD_RENDER))
			{
				fprintf(gpFile, "Could not load glyph\n");
				fflush(gpFile);
				continue;
			}
			//generate texture
			unsigned int tex;
			glGenTextures(1, &tex);
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			Character character = { tex,
									vec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
									vec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
									static_cast<unsigned int>(face->glyph->advance.x) };
			characters.insert(std::pair<char, Character>(c, character));
		}
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	FT_Done_Face(face);
	FT_Done_FreeType(ft);

}

void Initialize(void)
{
	//function prototype
	void Resize(int, int);
	GLuint loadTexture(void);
	void initFreetype(void);

	//variable declaration
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;
	GLuint quadVertexShaderObject;
	GLuint quadFragmentShaderObject;
	GLuint textVertexShaderObject;
	GLuint textFragmentShaderObject;

	//code

	//cuda
	int devCount;

	cudaResult = cudaGetDeviceCount(&devCount);
	if (cudaResult != cudaSuccess)
	{
		fprintf(gpFile, "cudaGetDeviceCount() unsuccessfull\n");
		Uninitialize();
		exit(EXIT_FAILURE);
	}
	else if (devCount == 0)
	{
		fprintf(gpFile, "devCount = 0\n");
		Uninitialize();
		exit(EXIT_FAILURE);
	}
	else
	{
		cudaSetDevice(0);
	}

	ghdc = GetDC(ghwnd);

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

	//struct pfd
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);

	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, ("ChoosePixelFormat() Failed.\n"));
		DestroyWindow(ghwnd);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, ("SetPixelFormat() Failed.\n"));
		DestroyWindow(ghwnd);
	}

	ghrc = wglCreateContext(ghdc);

	if (ghrc == NULL)
	{
		fprintf(gpFile, ("wglCreateContext() Failed.\n"));
		DestroyWindow(ghwnd);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, ("wglMakeCurrent() Failed\n"));
		DestroyWindow(ghwnd);
	}

	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//OpenGL related LOG

	fprintf(gpFile, "OpenGL VENDOR : %s\n", glGetString(GL_VENDOR));
	fprintf(gpFile, "OpenGL RENDERER : %s\n", glGetString(GL_RENDERER));
	fprintf(gpFile, "OpenGL VERSION : %s\n", glGetString(GL_VERSION));
	fprintf(gpFile, "GLSL[Graphics Library Shading Language] VERSION : %s\n\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	fprintf(gpFile, "EXTENTIONS : \n");

	//OpenGL Enabled Extensions

	GLint numExt;

	glGetIntegerv(GL_NUM_EXTENSIONS, &numExt);

	for (int i = 0; i < numExt; i++)
	{
		fprintf(gpFile, "%s\n", glGetStringi(GL_EXTENSIONS, i));
	}

	/****INIT FREETYPE****/
	initFreetype();

	/********SHADERS********/

#pragma region fullscreen_quad_shader

	/*****QUAD VERTEX SHADER*****/

	//create shader
	quadVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//provide source code to vertex shader
	const GLchar* quadVertexShaderSourceCode =
		"#version 450 core \n" \
		"\n" \
		"in vec4 vPosition; \n" \
		"in vec3 vColor; \n" \
		"in vec2 vTexcoord; \n" \
		"out vec3 out_color; \n" \
		"out vec2 out_texcoord; \n" \
		"void main(void) \n" \
		"{ \n" \
		"	out_color = vColor; \n" \
		"	out_texcoord = vTexcoord; \n" \
		"	gl_Position = vPosition; \n" \
		"} \n";

	glShaderSource(quadVertexShaderObject, 1, (const GLchar**)&quadVertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(quadVertexShaderObject);

	//vertex shader compilation error checking
	GLint infoLogLength = 0;
	GLint shaderCompiledStatus = 0;
	char* szBuffer = NULL;

	glGetShaderiv(quadVertexShaderObject, GL_COMPILE_STATUS, &shaderCompiledStatus);

	if (shaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(quadVertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(quadVertexShaderObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nQuad Vertex Shader Compilation Log : %s\n", szBuffer);
				fflush(gpFile);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}

		}
	}


	/*****FRAGMENT SHADER*****/

	//create shader
	quadFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//provide source code to fragment shader
	const GLchar* quadFragmentShaderSourceCode =
		"#version 450 core \n" \
		"\n" \
		"in vec3 out_color; \n" \
		"in vec2 out_texcoord; \n" \
		"uniform float u_color_alpha; \n" \
		"uniform sampler2D u_tex; \n" \
		"out vec4 FragColor; \n" \
		"void main(void) \n" \
		"{ \n" \
		/*"	FragColor = vec4(0.0, 1.0, 1.0, 1.0); \n" \*/
		/*"	FragColor = vec4(out_color, color_alpha_uniform); \n" \*/
		"	FragColor = vec4(texture(u_tex, out_texcoord).rgb, u_color_alpha); \n" \
		"} \n";

	glShaderSource(quadFragmentShaderObject, 1, (const GLchar**)&quadFragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(quadFragmentShaderObject);

	//fragment shader compilation error checking
	infoLogLength = 0;
	shaderCompiledStatus = 0;
	szBuffer = NULL;

	glGetShaderiv(quadFragmentShaderObject, GL_COMPILE_STATUS, &shaderCompiledStatus);

	if (shaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(quadFragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(quadFragmentShaderObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nQuad Fragment Shader Compilation Log : %s\n", szBuffer);
				fflush(gpFile);
				free(szBuffer);
				DestroyWindow(ghwnd);

			}
		}
	}


	/*****SHADER PROGRAM*****/

	//create
	quadShaderProgramObject = glCreateProgram();

	//attach vertex shader to shader program
	glAttachShader(quadShaderProgramObject, quadVertexShaderObject);

	//attach fragment shader to shader program
	glAttachShader(quadShaderProgramObject, quadFragmentShaderObject);

	//pre-linking binding
	glBindAttribLocation(quadShaderProgramObject, SPK_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(quadShaderProgramObject, SPK_ATTRIBUTE_COLOR, "vColor");
	glBindAttribLocation(quadShaderProgramObject, SPK_ATTRIBUTE_TEXCOORD, "vTexcoord");

	//link shader
	glLinkProgram(quadShaderProgramObject);

	//shader linking error checking
	infoLogLength = 0;
	GLint shaderProgramLinkStatus;
	szBuffer = NULL;

	glGetProgramiv(quadShaderProgramObject, GL_LINK_STATUS, &shaderProgramLinkStatus);

	if (shaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(quadShaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);

			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(quadShaderProgramObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nQuad Shader Program Link Log : %s\n", szBuffer);
				fflush(gpFile);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//get MVP uniform location
	color_alpha_uniform = glGetUniformLocation(quadShaderProgramObject, "u_color_alpha");
	sampler_tex_uniform = glGetUniformLocation(quadShaderProgramObject, "u_tex");

#pragma endregion

#pragma region text_shader

	/*****TEXT VERTEX SHADER*****/

	//create shader
	textVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//provide source code to vertex shader
	const GLchar* textVertexShaderSourceCode =
		"#version 450 core \n" \
		"\n" \
		"in vec4 vPosition; \n" \
		"out vec2 texcoords; \n" \
		"uniform mat4 u_mvpMatrix; \n" \
		"void main(void) \n" \
		"{ \n" \
		"	gl_Position = u_mvpMatrix * vec4(vPosition.xy, 0.0, 1.0); \n" \
		"	texcoords = vPosition.zw; \n" \
		"} \n";

	glShaderSource(textVertexShaderObject, 1, (const GLchar**)&textVertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(textVertexShaderObject);

	//vertex shader compilation error checking
	infoLogLength = 0;
	shaderCompiledStatus = 0;
	szBuffer = NULL;

	glGetShaderiv(textVertexShaderObject, GL_COMPILE_STATUS, &shaderCompiledStatus);

	if (shaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(textVertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(textVertexShaderObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nText Vertex Shader Compilation Log : %s\n", szBuffer);
				fflush(gpFile);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}

		}
	}


	/*****FRAGMENT SHADER*****/

	//create shader
	textFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//provide source code to fragment shader
	const GLchar* textFragmentShaderSourceCode =
		"#version 450 core \n" \
		"\n" \
		"in vec2 texcoords; \n" \
		"out vec4 FragColor; \n" \
		"uniform sampler2D text; \n" \
		"uniform vec4 textColor; \n" \
		"void main(void) \n" \
		"{ \n" \
		"	vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, texcoords).r); \n" \
		"	FragColor = textColor * sampled; \n" \
		"} \n";

	glShaderSource(textFragmentShaderObject, 1, (const GLchar**)&textFragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(textFragmentShaderObject);

	//fragment shader compilation error checking
	infoLogLength = 0;
	shaderCompiledStatus = 0;
	szBuffer = NULL;

	glGetShaderiv(textFragmentShaderObject, GL_COMPILE_STATUS, &shaderCompiledStatus);

	if (shaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(textFragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(textFragmentShaderObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nText Fragment Shader Compilation Log : %s\n", szBuffer);
				fflush(gpFile);
				free(szBuffer);
				DestroyWindow(ghwnd);

			}
		}
	}


	/*****SHADER PROGRAM*****/

	//create
	textShaderProgramObject = glCreateProgram();

	//attach vertex shader to shader program
	glAttachShader(textShaderProgramObject, textVertexShaderObject);

	//attach fragment shader to shader program
	glAttachShader(textShaderProgramObject, textFragmentShaderObject);

	//pre-linking binding
	glBindAttribLocation(textShaderProgramObject, SPK_ATTRIBUTE_POSITION, "vPosition");

	//link shader
	glLinkProgram(textShaderProgramObject);

	//shader linking error checking
	infoLogLength = 0;
	shaderProgramLinkStatus;
	szBuffer = NULL;

	glGetProgramiv(textShaderProgramObject, GL_LINK_STATUS, &shaderProgramLinkStatus);

	if (shaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(textShaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);

			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(textShaderProgramObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nShader Program Link Log : %s\n", szBuffer);
				fflush(gpFile);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//get MVP uniform location
	mvpMatrixUniformText = glGetUniformLocation(textShaderProgramObject, "u_mvpMatrix");
	textSamplerUniform = glGetUniformLocation(textShaderProgramObject, "text");
	textColorUniform = glGetUniformLocation(textShaderProgramObject, "textColor");


#pragma endregion

#pragma region mandelbrot_shader

	/*****VERTEX SHADER*****/

	//create shader
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//provide source code to vertex shader
	const GLchar* vertexShaderSourceCode =
		"#version 450 core \n" \
		"\n" \

		"in vec4 vPosition; \n" \
		"in vec2 vTexcoord; \n" \
		"out vec2 out_texcoord; \n" \

		"uniform mat4 u_mvp_matrix; \n" \

		"void main(void) \n" \
		"{ \n" \
		
		"	out_texcoord = vTexcoord; \n" \
		/*"	out_color = vPosition * 0.5 + 0.5; \n" \
		"	out_color.z = 0.0; \n" \*/
		"	gl_Position = u_mvp_matrix * vPosition; \n" \
		
		"} \n";

	glShaderSource(vertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(vertexShaderObject);

	//vertex shader compilation error checking
	infoLogLength = 0;
	shaderCompiledStatus = 0;
	szBuffer = NULL;

	glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &shaderCompiledStatus);

	if (shaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nVertex Shader Compilation Log : %s\n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}

		}
	}


	/*****FRAGMENT SHADER*****/

	//create shader
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//provide source code to fragment shader
	const GLchar* fragmentShaderSourceCode =
		"#version 450 core \n" \
		"\n" \

		"in vec2 out_texcoord; \n" \
		"out vec4 Fragcolor; \n" \
		"uniform sampler2D u_tex_sampler;" \

		"void main(void) \n" \
		"{ \n" \

		/*"	Fragcolor = vec4(0.0, 1.0, 1.0, 1.0); \n" \*/
		"	Fragcolor = texture(u_tex_sampler, out_texcoord); \n" \
		"	Fragcolor.a = 1.0; \n" \

		"} \n";

	glShaderSource(fragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(fragmentShaderObject);

	//fragment shader compilation error checking
	infoLogLength = 0;
	shaderCompiledStatus = 0;
	szBuffer = NULL;

	glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &shaderCompiledStatus);

	if (shaderCompiledStatus == GL_FALSE)
	{
		glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nFragment Shader Compilation Log : %s\n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);

			}
		}
	}


	/*****SHADER PROGRAM*****/

	//create
	shaderProgramObject = glCreateProgram();

	//attach vertex shader to shader program
	glAttachShader(shaderProgramObject, vertexShaderObject);

	//attach fragment shader to shader program
	glAttachShader(shaderProgramObject, fragmentShaderObject);

	//pre-linking binding
	glBindAttribLocation(shaderProgramObject, SPK_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(shaderProgramObject, SPK_ATTRIBUTE_TEXCOORD, "vTexcoord");

	//link shader
	glLinkProgram(shaderProgramObject);

	//shader linking error checking
	infoLogLength = 0;
	shaderProgramLinkStatus;
	szBuffer = NULL;

	glGetProgramiv(shaderProgramObject, GL_LINK_STATUS, &shaderProgramLinkStatus);

	if (shaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(shaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 0)
		{
			szBuffer = (char*)malloc(infoLogLength);

			if (szBuffer != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(shaderProgramObject, infoLogLength, &written, szBuffer);
				fprintf(gpFile, "\nShader Program Link Log : %s\n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//get MVP uniform location
	texSamplerUniform = glGetUniformLocation(shaderProgramObject, "u_tex_sampler");
	mvpUniform = glGetUniformLocation(shaderProgramObject, "u_mvp_matrix");

#pragma endregion

	
	//vertices array declaration
	const GLfloat squareVertices[] = {
	   1.0f,  1.0f, 0.0f,
	  -1.0f,  1.0f, 0.0f,
	  -1.0f, -1.0f, 0.0f,
	   1.0f, -1.0f, 0.0f
	};

	const GLfloat squareTexcoords[] = {
	    1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	const GLfloat squareColors[] = {
		0.0f, 0.0f, 0.1f,
		0.0f, 0.0f, 0.1f,
		0.0f, 0.0f, 0.1f,
		0.0f, 0.0f, 0.1f
	};

	//vao & vbo
	//mandelbrot 
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo_pos_quad);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_quad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(SPK_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SPK_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_tex_quad);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_tex_quad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareTexcoords), squareTexcoords, GL_STATIC_DRAW);
	glVertexAttribPointer(SPK_ATTRIBUTE_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SPK_ATTRIBUTE_TEXCOORD);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vbo_col_quad);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_col_quad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareColors), squareColors, GL_STATIC_DRAW);
	glVertexAttribPointer(SPK_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SPK_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);


	//text
	glGenVertexArrays(1, &vao_text);
	glBindVertexArray(vao_text);

	glGenBuffers(1, &vbo_text);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(SPK_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
	glEnableVertexAttribArray(SPK_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	//depth
	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);
	
	//enable blending for text
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//Audio
	if (PlaySong == false)
	{
		PlaySound(MAKEINTRESOURCE(MYAUDIO), GetModuleHandle(NULL), SND_RESOURCE | SND_ASYNC);
		PlaySong = true;
	}

	//interop texture
	mandelTextureCPU = loadTexture();
	mandelTextureGPU = loadTexture();

	cudaResult = cudaGraphicsGLRegisterImage(&cuda_graphics_resource_tex, mandelTextureGPU, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
	if (cudaResult != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterImage() for vbo_tex unsuccessfull\n");
		Uninitialize();
		exit(EXIT_FAILURE);
	}

	cudaResult = cudaMalloc((void**)&gpu_tex, MAX_WIDTH * MAX_HEIGHT * 4);
	if (cudaResult != cudaSuccess)
	{
		fprintf(gpFile, "cudaMalloc() for gpu_tex unsuccessfull\n");
		Uninitialize();
		exit(EXIT_FAILURE);
	}

	//variable initialization
	textAlphaVal.alpha[0] = 1.0f;
	textAlphaVal.alpha[1] = 1.0f;
	textAlphaVal.alpha[2] = 0.0f;
	textAlphaVal.alpha[3] = 1.0f;
	textAlphaVal.alpha[4] = 1.0f;
	textAlphaVal.alpha[5] = 1.0f;
	textAlphaVal.alpha[6] = 1.0f;

	//load textures
	tex1_amc = LoadImageAsTexture("res/textures/01_amc.jpg");
	tex2_grp = LoadImageAsTexture("res/textures/02_grp.jpg");
	tex3_mandelbrot = LoadImageAsTexture("res/textures/03_mandelbrot.jpg");
	tex4_sk = LoadImageAsTexture("res/textures/04_sk.jpg");
	tex5_sir = LoadImageAsTexture("res/textures/05_sir.jpg");
	tex6_tech = LoadImageAsTexture("res/textures/06_tech.jpg");
	tex7_ref = LoadImageAsTexture("res/textures/07_ref.jpg");
	tex8_splTy = LoadImageAsTexture("res/textures/08_splTy.jpg");
	tex9_ty = LoadImageAsTexture("res/textures/09_ty.jpg");

	//SetClearColor
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//set perspective matrix to identity matrix
	perspectiveProjectionMatrix = mat4::identity();

	//set fps to system
	wglSwapIntervalEXT(0);   //0 --> will extend beyond 60

	Resize(WIN_WIDTH, WIN_HEIGHT);
}

GLuint LoadImageAsTexture(const char* path)
{
	//variable declarations    
	int width, height;
	unsigned char* imageData = NULL;
	int nrComponents;

	//code    
	imageData = SOIL_load_image(path, &width, &height, &nrComponents, 0);

	if (imageData)
	{
		GLenum format;
		if (nrComponents == 1)
			format = GL_RED;
		else if (nrComponents == 3)
			format = GL_RGB;
		else if (nrComponents == 4)
			format = GL_RGBA;

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, imageData);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT); // for this tutorial: use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes texels from next repeat 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, format == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		SOIL_free_image_data(imageData);

	}
	else
	{
		SOIL_free_image_data(imageData);
	}

	return(textureID);

}

GLuint loadTexture(void)
{
	//launch the CPU kernel
	//launchCPUKernel();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glGenTextures(1, &texImage);

	glBindTexture(GL_TEXTURE_2D, texImage);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, MAX_WIDTH, MAX_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	return(texImage);

}

void Resize(int width, int height)
{
	//code
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void launchCPUKernel(vec4 cpu_outerColor1, vec4 cpu_outerColor2)
{
	vec4 innerColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec4 outerColor1 = cpu_outerColor1;
	vec4 outerColor2 = cpu_outerColor2;
	vec4 color;

	int offset;
	float real;
	float imag;
	float cReal;
	float cImag;
	float dist;
	int iter;
	float tmp_real;
	float tmp_imag;

	float fx, fy;

	for (int x = 0; x < MAX_WIDTH; x++)
	{
		for (int y = 0; y < MAX_HEIGHT; y++)
		{
			//to convert 2 loops in 1D
			offset = x + y * MAX_WIDTH;

			fx = (float)x / (float)MAX_WIDTH;
			fy = (float)y / (float)MAX_HEIGHT;

			real = ((float)MAX_WIDTH / (float)MAX_HEIGHT) * fx * zoom + xCenter;
			//real = (1.77) * fx * zoom + xCenter;
			imag = fy * zoom + yCenter;  

			cReal = real;  
			cImag = imag;  
		
			for(iter = 0; (iter < maxIterations); iter++)  
			{  
				tmp_real = (real * real) - (imag * imag);
				tmp_imag = (2.0 * real * imag);

				real = tmp_real + cReal;
				imag = tmp_imag + cImag;

				dist = (real * real) + (imag * imag);

				if (dist > 16.0)
					break;
			}  

			//pixel color calculation

			//if (iter == maxIterations)
			if (dist < 4.0)
				color = innerColor;
			else
				color = LERP(outerColor1, outerColor2, FRACT(iter * 0.013f));   //0.015f

			pos[offset * 4 + 0] = 255 * color[0];
			pos[offset * 4 + 1] = 255 * color[1];
			pos[offset * 4 + 2] = 255 * color[2];
			pos[offset * 4 + 3] = 255 * color[3];
		}
	}

}

void Display(void)
{
	//prototype
	void RenderText(GLuint, std::string, GLfloat, GLfloat, GLfloat, vec4);

	currentTime += 0.01f;

	//code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//intro scene

	if (spaceKey_scene == 1)
	{
		glUseProgram(quadShaderProgramObject);

		glUniform1f(color_alpha_uniform, quadAlphaVal);
		
		glActiveTexture(GL_TEXTURE0);

		if (currentTexture == 1)
			glBindTexture(GL_TEXTURE_2D, tex1_amc);
		else if(currentTexture==2)
			glBindTexture(GL_TEXTURE_2D, tex2_grp);
		else if(currentTexture==3)
			glBindTexture(GL_TEXTURE_2D, tex3_mandelbrot);

		glUniform1i(sampler_tex_uniform, 0);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		//unbind vao
		glBindVertexArray(0);

		glUseProgram(0);
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//render mandelbrot

	else if (spaceKey_scene == 2)
	{
		//start using OpenGL program object
		glUseProgram(shaderProgramObject);

		mat4 translateMatrix;
		mat4 scaleMatrix;
		mat4 modelViewMatrix;
		mat4 modelViewProjectionMatrix;

		translateMatrix = mat4::identity();
		modelViewMatrix = mat4::identity();
		modelViewProjectionMatrix = mat4::identity();

		translateMatrix = translate(0.0f, 0.0f, -3.0f);
		modelViewMatrix = translateMatrix;
		//modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;  //pre-multiplication of matrices
		modelViewProjectionMatrix = mat4::identity();
		glUniformMatrix4fv(mvpUniform, 1, GL_FALSE, modelViewProjectionMatrix);

		//bind vao
		glBindVertexArray(vao);

		if (bOnGPU == true)
		{
			//launchCUDAKernel( maxWidth, maxHeight, zoom, xCenter, yCenter, maxIterations);
			launchCUDAKernel(gpu_tex, maxWidth, maxHeight, zoom, xCenter, yCenter, maxIterations, cuOuterColor1, cuOuterColor2);

			//pos
			cudaResult = cudaGraphicsMapResources(1, &cuda_graphics_resource_tex, 0);
			if (cudaResult != cudaSuccess)
			{
				fprintf(gpFile, "cudaGraphicsMapResources() for pos unsuccessfull\n");
				Uninitialize();
				exit(EXIT_FAILURE);
			}

			cudaArray_t pPos;
			size_t numBytes;

			//cudaResult = cudaGraphicsResourceGetMappedPointer((void**)&pPos, &numBytes, cuda_graphics_resource_tex);
			cudaResult = cudaGraphicsSubResourceGetMappedArray(&pPos, cuda_graphics_resource_tex, 0, 0);
			if (cudaResult != cudaSuccess)
			{
				fprintf(gpFile, "cudaGraphicsResourceGetMappedPointer() for pos unsuccessfull\n");
				Uninitialize();
				exit(EXIT_FAILURE);
			}

			cudaResult = cudaMemcpyToArray(pPos, 0, 0, gpu_tex, MAX_WIDTH * MAX_HEIGHT * 4, cudaMemcpyDeviceToDevice);
			if (cudaResult != cudaSuccess)
			{
				fprintf(gpFile, "cudaMemcpyToArray() for pos unsuccessfull\n");
				Uninitialize();
				exit(EXIT_FAILURE);
			}

			cudaResult = cudaGraphicsUnmapResources(1, &cuda_graphics_resource_tex, 0);
			if (cudaResult != cudaSuccess)
			{
				fprintf(gpFile, "cudaGraphicsUnmapResources() for pos unsuccessfull\n");
				Uninitialize();
				exit(EXIT_FAILURE);
			}

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, mandelTextureGPU);
			glUniform1i(GL_TEXTURE_2D, 0);

		}
		else
		{
			//call the CPU kernel func
			launchCPUKernel(cpuOuterColor1, cpuOuterColor2);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, mandelTextureCPU);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MAX_WIDTH, MAX_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pos);
			glUniform1i(GL_TEXTURE_2D, 0);

		}

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		//unbind vao
		glBindVertexArray(0);


		//stop using OpenGL program object
		glUseProgram(0);
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//end credits

	else if (spaceKey_scene == 3)
	{
		glUseProgram(quadShaderProgramObject);

		glUniform1f(color_alpha_uniform, quadAlphaVal);

		glActiveTexture(GL_TEXTURE0);

		if (currentTexture == 4)
			glBindTexture(GL_TEXTURE_2D, tex4_sk);
		else if (currentTexture == 5)
			glBindTexture(GL_TEXTURE_2D, tex5_sir);
		else if (currentTexture == 6)
			glBindTexture(GL_TEXTURE_2D, tex6_tech);
		else if (currentTexture == 7)
			glBindTexture(GL_TEXTURE_2D, tex7_ref);
		else if (currentTexture == 8)
			glBindTexture(GL_TEXTURE_2D, tex8_splTy);
		else if (currentTexture == 9)
			glBindTexture(GL_TEXTURE_2D, tex9_ty);

		glUniform1i(sampler_tex_uniform, 0);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		//unbind vao
		glBindVertexArray(0);

		glUseProgram(0);
	}



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//render text

#pragma region text

	if (spaceKey_scene == 1)
	{
		/*RenderText(textShaderProgramObject, "ASTROMEDICOMP'S", 0.0f, 0.0f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[0]));
		RenderText(textShaderProgramObject, "Domain Group", 0.0f, -0.1f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[0]));
		RenderText(textShaderProgramObject, "presents", 0.0f, -0.2f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[0]));

		RenderText(textShaderProgramObject, "MANDELBROT SET", 0.0f, 0.0f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[1]));*/
	}
	else if (spaceKey_scene == 2)
	{
		if (bOnGPU == true)
		{
			RenderText(textShaderProgramObject, "Mandelbrot on GPU", -0.9f, 0.85f, 0.001f, vec4(0.16f, 0.71f, 0.0f, 1.0f));
			//RenderText(textShaderProgramObject, "GPU - NVIDIA GeForce RTX 3060 Ti", -0.9f, 0.85f, 0.0009f, vec4(0.16f, 0.71f, 0.0f, 1.0f));
			RenderText(textShaderProgramObject, fps_str, -0.9f, 0.75f, 0.0009f, vec4(0.16f, 0.71f, 0.0f, 1.0f));
			RenderText(textShaderProgramObject, ms_str, -0.9f, 0.65f, 0.0009f, vec4(0.16f, 0.71f, 0.0f, 1.0f));
			RenderText(textShaderProgramObject, iter_str, -0.9f, 0.55f, 0.0009f, vec4(0.16f, 0.71f, 0.0f, 1.0f));
		}

		else
		{
			RenderText(textShaderProgramObject, "Mandelbrot on CPU", -0.9f, 0.85f, 0.001f, vec4(0.93f, 0.68f, 0.0f, textAlphaVal.alpha[2]));
			//RenderText(textShaderProgramObject, "CPU - Intel(R) Core(TM) i7-11700", -0.9f, 0.85f, 0.0009f, vec4(0.93f, 0.68f, 0.0f, textAlphaVal.alpha[2]));
			RenderText(textShaderProgramObject, fps_str, -0.9f, 0.75f, 0.0009f, vec4(0.93f, 0.68f, 0.0f, textAlphaVal.alpha[2]));
			RenderText(textShaderProgramObject, ms_str, -0.9f, 0.65f, 0.0009f, vec4(0.93f, 0.68f, 0.0f, textAlphaVal.alpha[2]));
			RenderText(textShaderProgramObject, iter_str, -0.9f, 0.55f, 0.0009f, vec4(0.93f, 0.68f, 0.0f, textAlphaVal.alpha[2]));
		}


		if (iKeyPressed == true)
			RenderText(textShaderProgramObject, "\' I \' key pressed to change iterations", -0.3f, -0.9f, 0.0009f, vec4(1.0f, 1.0f, 1.0f, 1.0f));

		if (sKeyPressed == 1)
			RenderText(textShaderProgramObject, "\' Z \' key pressed to zoom", -0.25f, -0.9f, 0.0009f, vec4(1.0f, 1.0f, 1.0f, 1.0f));

		
	}
	else if (spaceKey_scene == 3)
	{
		/*RenderText(textShaderProgramObject, "Guided By", 0.0f, 0.0f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[3]));
		RenderText(textShaderProgramObject, "Dr. Vijay D. Gokhale Sir", 0.0f, -0.1f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[3]));
		

		RenderText(textShaderProgramObject, "Technology Used", 0.0f, 0.0f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[4]));
		RenderText(textShaderProgramObject, "OS - Windows", 0.0f, -0.1f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[4]));
		RenderText(textShaderProgramObject, "Rendering - OpenGL", 0.0f, -0.2f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[4]));
		RenderText(textShaderProgramObject, "HPP - CUDA", 0.0f, -0.3f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[4]));
		
		RenderText(textShaderProgramObject, "Reference", 0.0f, -0.6f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[5]));
		RenderText(textShaderProgramObject, "CUDA Samples", 0.0f, -0.7f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[5]));
		
		RenderText(textShaderProgramObject, "Thank You", 0.0f, 0.0f, 0.001f, vec4(1.0f, 1.0f, 1.0f, textAlphaVal.alpha[6]));*/
	}

#pragma endregion
	
	//glDisable(GL_BLEND);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	SwapBuffers(ghdc);
}

void Update(void)
{
	//code

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//SCENE 1

	if (spaceKey_scene == 1)
	{
		static int animFlow = 1;

		switch (animFlow)
		{
		case 1:
			quadAlphaVal += 0.00005f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 2:
			quadAlphaVal -= 0.00005f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 2;
				animFlow++;
			}
			break;

		case 3:
			quadAlphaVal += 0.00005f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 4:
			quadAlphaVal -= 0.00005f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 3;
				animFlow++;
			}
			break;

		case 5:
			quadAlphaVal += 0.000025f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 6:
			quadAlphaVal -= 0.000025f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 4;
				spaceKey_scene = 2;
				//animFlow++;
			}
			break;
		}
		
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//SCENE 2

	else if (spaceKey_scene == 2)
	{
		textAlphaVal.alpha[2] += 0.1f;
		if (textAlphaVal.alpha[2] >= 1.0f)
		{
			textAlphaVal.alpha[2] = 1.0f;
		}

		zoom = LERP(3.0f, 0.020003f, dt);
		xCenter = LERP(-3.0f, -0.320003f, dt);
		yCenter = LERP(-1.5, 0.649999, dt);

		if (sKeyPressed == 1)
		{
			dt += 0.05f;

			if (dt >= 1.0)
				dt = 1.0f;
		}
		else if (sKeyPressed == 0)
		{
			dt -= 0.05f;

			if (dt <= 0.0)
				dt = 0.0f;
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//SCENE 3

	else if (spaceKey_scene == 3)
	{
		static int animFlow = 1;

		switch (animFlow)
		{
		case 1:
			quadAlphaVal += 0.00006f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 2:
			quadAlphaVal -= 0.00006f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 5;
				animFlow++;
			}
			break;

		case 3:
			quadAlphaVal += 0.00006f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 4:
			quadAlphaVal -= 0.00006f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 6;
				animFlow++;
			}
			break;

		case 5:
			quadAlphaVal += 0.00006f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 6:
			quadAlphaVal -= 0.00006f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 7;
				animFlow++;
			}
			break;

		case 7:
			quadAlphaVal += 0.00006f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 8:
			quadAlphaVal -= 0.00006f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 8;
				animFlow++;
			}
			break;

		case 9:
			quadAlphaVal += 0.00006f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				animFlow++;
			}
			break;

		case 10:
			quadAlphaVal -= 0.00006f;
			if (quadAlphaVal <= 0.0f)
			{
				quadAlphaVal = 0.0f;
				currentTexture = 9;
				animFlow++;
			}
			break;

		case 11:
			quadAlphaVal += 0.00006f;
			if (quadAlphaVal >= 1.0f)
			{
				quadAlphaVal = 1.0f;
				//animFlow++;
			}
			break;

		//case 8:
		//	quadAlphaVal -= 0.00003f;
		//	if (quadAlphaVal <= 0.0f)
		//	{
		//		quadAlphaVal = 0.0f;
		//		//currentTexture = 4;
		//		//spaceKey_scene = 2;
		//		//animFlow++;
		//	}
		//	break;
		}
	}

#pragma region with_freetype

	//if (spaceKey_scene == 1)
	//{
	//	static int animFlow = 1;

	//	switch (animFlow)
	//	{
	//	case 1:
	//		textAlphaVal.alpha[0] += 0.0001f;
	//		if (textAlphaVal.alpha[0] >= 1.0f)
	//		{
	//			textAlphaVal.alpha[0] = 1.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 2:
	//		textAlphaVal.alpha[0] -= 0.0001f;
	//		if (textAlphaVal.alpha[0] <= 0.0f)
	//		{
	//			textAlphaVal.alpha[0] = 0.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 3:
	//		textAlphaVal.alpha[1] += 0.0001f;
	//		if (textAlphaVal.alpha[1] >= 1.0f)
	//		{
	//			textAlphaVal.alpha[1] = 1.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 4:
	//		textAlphaVal.alpha[1] -= 0.0001f;
	//		if (textAlphaVal.alpha[1] <= 0.0f)
	//		{
	//			textAlphaVal.alpha[1] = 0.0f;
	//		}

	//		quadAlphaVal -= 0.0001f;
	//		if (quadAlphaVal <= 0.0f)
	//		{
	//			quadAlphaVal = 0.0f;
	//			animFlow++;
	//			spaceKey_scene = 2;
	//		}
	//		break;
	//		
	//	}

	//	
	//}

	//else if (spaceKey_scene == 2)
	//{
	//	textAlphaVal.alpha[2] += 0.1f;
	//	if (textAlphaVal.alpha[2] >= 1.0f)
	//	{
	//		textAlphaVal.alpha[2] = 1.0f;
	//	}

	//	zoom = LERP(3.0f, 0.020003f, dt);
	//	xCenter = LERP(-3.0f, -0.320003f, dt);
	//	yCenter = LERP(-1.5, 0.649999, dt);

	//	if (sKeyPressed == 1)
	//	{
	//		dt += 0.05f;

	//		if (dt >= 1.0)
	//			dt = 1.0f;
	//	}
	//	else if (sKeyPressed == 0)
	//	{
	//		dt -= 0.05f;

	//		if (dt <= 0.0)
	//			dt = 0.0f;
	//	}
	//}
	//else if (spaceKey_scene == 3)
	//{
	//	static int animFlow = 1;

	//	switch (animFlow)
	//	{
	//	case 1:
	//		quadAlphaVal += 0.0001f;
	//		if (quadAlphaVal >= 1.0f)
	//		{
	//			quadAlphaVal = 1.0f;
	//			animFlow++;
	//		}

	//	case 2:
	//		textAlphaVal.alpha[3] += 0.0001f;
	//		if (textAlphaVal.alpha[3] >= 1.0f)
	//		{
	//			textAlphaVal.alpha[3] = 1.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 3:
	//		textAlphaVal.alpha[3] -= 0.0001f;
	//		if (textAlphaVal.alpha[3] <= 0.0f)
	//		{
	//			textAlphaVal.alpha[3] = 0.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 4:
	//		textAlphaVal.alpha[4] += 0.0001f;
	//		if (textAlphaVal.alpha[4] >= 1.0f)
	//		{
	//			textAlphaVal.alpha[4] = 1.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 5:
	//		textAlphaVal.alpha[5] += 0.0001f;
	//		if (textAlphaVal.alpha[5] >= 1.0f)
	//		{
	//			textAlphaVal.alpha[5] = 1.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 6:
	//		textAlphaVal.alpha[4] -= 0.0001f;
	//		if (textAlphaVal.alpha[4] <= 0.0f)
	//		{
	//			textAlphaVal.alpha[4] = 0.0f;
	//		}

	//		textAlphaVal.alpha[5] -= 0.0001f;
	//		if (textAlphaVal.alpha[5] <= 0.0f)
	//		{
	//			textAlphaVal.alpha[5] = 0.0f;
	//			animFlow++;
	//		}
	//		break;

	//	case 7:
	//		textAlphaVal.alpha[6] += 0.0001f;
	//		if (textAlphaVal.alpha[6] >= 1.0f)
	//		{
	//			textAlphaVal.alpha[6] = 1.0f;
	//			//animFlow++;
	//		}
	//		break;
	//	}
	//}

#pragma endregion
	
}

void RenderText(GLuint shader, std::string text, GLfloat x, GLfloat y, GLfloat scale, vec4 color)
{
	mat4 translateMatrix;
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;

	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	glUseProgram(shader);
	glUniform4f(textColorUniform, color[0], color[1], color[2], color[3]);

	//translateMatrix = translate(0.0f, 0.0f, -1.0f);
	modelViewMatrix = translateMatrix;
	//modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;  //pre-multiplication of matrices
	modelViewProjectionMatrix = mat4::identity();
	glUniformMatrix4fv(mvpMatrixUniformText, 1, GL_FALSE, modelViewProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindVertexArray(vao_text);

	std::string::const_iterator c;
	for (c = text.begin(); c != text.end(); c++)
	{
		Character ch = characters[*c];

		float xpos = x + ch.bearing[0] * scale;
		float ypos = y - (ch.size[1] - ch.bearing[1]) * scale * 1.7f;  //(1.7 --> aspect ratio) [jugaad]

		float w = ch.size[0] * scale;
		float h = ch.size[1] * scale * 1.7f;    //(1.7 --> aspect ratio) [jugaad]

		float vertices[6][4] = {
			{xpos, ypos + h, 0.0f, 0.0f},
			{xpos, ypos, 0.0f, 1.0f},
			{xpos + w, ypos, 1.0f, 1.0f},

			{xpos, ypos + h, 0.0f, 0.0f},
			{xpos + w, ypos, 1.0f, 1.0f},
			{xpos + w, ypos + h, 1.0f, 0.0f}
		};

		glBindTexture(GL_TEXTURE_2D, ch.textTextureID);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_text);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		x += (ch.advance >> 6) * scale;

	}

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);
}

void Uninitialize(void)
{
	//code
	if (gbFullscreen == true)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

		SetWindowLong(ghwnd, GWL_STYLE, (dwStyle | WS_OVERLAPPEDWINDOW));
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED);

		ShowCursor(true);
	}

	if (PlaySong == true)
	{
		PlaySound(NULL, NULL, NULL);
	}


	//quad
	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}
	
	if (vbo_pos_quad)
	{
		glDeleteVertexArrays(1, &vbo_pos_quad);
		vbo_pos_quad = 0;
	}
	if (vbo_tex_quad)
	{
		glDeleteVertexArrays(1, &vbo_tex_quad);
		vbo_tex_quad = 0;
	}
	if (vbo_col_quad)
	{
		glDeleteVertexArrays(1, &vbo_col_quad);
		vbo_col_quad = 0;
	}

	//text
	if (vao_text)
	{
		glDeleteVertexArrays(1, &vao_text);
		vao_text = 0;
	}

	if (vbo_text)
	{
		glDeleteBuffers(1, &vbo_text);
		vbo_text = 0;
	}

	//delete textures
	glDeleteTextures(1, &tex1_amc);
	glDeleteTextures(1, &tex2_grp);
	glDeleteTextures(1, &tex3_mandelbrot);
	glDeleteTextures(1, &tex4_sk);
	glDeleteTextures(1, &tex5_sir);
	glDeleteTextures(1, &tex6_tech);
	glDeleteTextures(1, &tex7_ref);
	glDeleteTextures(1, &tex8_splTy);
	glDeleteTextures(1, &tex9_ty);
	glDeleteTextures(1, &mandelTextureCPU);
	glDeleteTextures(1, &mandelTextureGPU);


	/*****SAFE SHADER CLEAN-UP*****/

	if (shaderProgramObject)
	{
		glUseProgram(shaderProgramObject);
		GLsizei shaderCount;
		glGetProgramiv(shaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint* pShaders = NULL;
		pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);

		glGetAttachedShaders(shaderProgramObject, shaderCount, &shaderCount, pShaders);

		for (GLsizei i = 0; i < shaderCount; i++)
		{
			glDetachShader(shaderProgramObject, pShaders[i]);
			glDeleteShader(pShaders[i]);
			pShaders[i] = 0;
		}
		free(pShaders);

		glDeleteProgram(shaderProgramObject);
		shaderProgramObject = 0;
		glUseProgram(0);

	}

	if (textShaderProgramObject)
	{
		glUseProgram(textShaderProgramObject);
		GLsizei shaderCount;
		glGetProgramiv(textShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint* pShaders = NULL;
		pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);

		glGetAttachedShaders(textShaderProgramObject, shaderCount, &shaderCount, pShaders);

		for (GLsizei i = 0; i < shaderCount; i++)
		{
			glDetachShader(textShaderProgramObject, pShaders[i]);
			glDeleteShader(pShaders[i]);
			pShaders[i] = 0;
		}
		free(pShaders);

		glDeleteProgram(textShaderProgramObject);
		textShaderProgramObject = 0;
		glUseProgram(0);

	}

	if (quadShaderProgramObject)
	{
		glUseProgram(quadShaderProgramObject);
		GLsizei shaderCount;
		glGetProgramiv(quadShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint* pShaders = NULL;
		pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);

		glGetAttachedShaders(quadShaderProgramObject, shaderCount, &shaderCount, pShaders);

		for (GLsizei i = 0; i < shaderCount; i++)
		{
			glDetachShader(quadShaderProgramObject, pShaders[i]);
			glDeleteShader(pShaders[i]);
			pShaders[i] = 0;
		}
		free(pShaders);

		glDeleteProgram(quadShaderProgramObject);
		quadShaderProgramObject = 0;
		glUseProgram(0);

	}

	//
	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	//
	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	//
	if (ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//
	if (gpFile)
	{
		fprintf(gpFile, ("\nLog File Closed Successfully, Program Completed Successfully.\n"));
		fclose(gpFile);
		gpFile = NULL;
	}
}

