#pragma once
#include <driver_types.h>
#include <cstdio>
#include <cuda_runtime_api.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/// <summary>
/// To check cuda errors
/// </summary>
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}