/**
* @file    2D Median filterusing NVIDIA CUDA
* @author  Institute for Photon Science and Synchrotron Radiation, Karlsruhe Institute of Technology
*
* @date    2015-2018
* @version 0.5.0
*
*
* @section LICENSE
*
* This program is copyrighted by the author and Institute for Photon Science and Synchrotron Radiation,
* Karlsruhe Institute of Technology, Karlsruhe, Germany;
*
* The current implemetation contains the following licenses:
*
* 1. TinyXml package:
*      Original code (2.0 and earlier )copyright (c) 2000-2006 Lee Thomason (www.grinninglizard.com). <www.sourceforge.net/projects/tinyxml>.
*      See src/utils/tinyxml.h for details.
*
*/

#include <device_launch_parameters.h>

#define __CUDACC__

#include <device_functions.h>
#include <math_functions.h>

#include "src/data_types/data_structs.h"

//#define IND(X, Y, Z) (((Z) * container_size.height + (Y)) * (container_size.pitch / sizeof(float)) + (X)) 
//#define SIND(X, Y, Z) ((((Z) + radius_2) * shared_block_size.y + ((Y) + radius_2)) * shared_block_size.x + ((X) + radius_2))

#define IND(X, Y) ((Y) * (container_size.pitch / sizeof(float)) + (X)) 
#define SIND(X, Y) ((((Y) + radius_2)) * shared_block_size.x + ((X) + radius_2))

__constant__ DataSize3 container_size;


extern __shared__ float shared[];

__device__ void sort(float* buffer, size_t length)
{
  for (int i = 0; i < length - 1; i++) {
    for (int k = 0; k < length - i - 1; k++) {
      if (buffer[k] > buffer[k + 1]) {
        float a = buffer[k];
        buffer[k] = buffer[k + 1];
        buffer[k + 1] = a;
      }
    }
  }
}

/* See a note about the thread block size in cuda_operation_median.cpp file.*/
extern "C" __global__ void median_2d(
  const float* input,
        size_t width,
        size_t height,
        size_t radius,
        float* output)
{
  int radius_2 = radius / 2;

  dim3 shared_block_size(
    blockDim.x + 2 * radius_2,
    blockDim.y + 2 * radius_2
   );

  dim3 global_id(
    blockDim.x * blockIdx.x + threadIdx.x,
    blockDim.y * blockIdx.y + threadIdx.y
   );

  /* Load data to the shared memoty */
  size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
  size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;

  /* Main area */
  shared[SIND(threadIdx.x, threadIdx.y)] = input[IND(global_x, global_y)];

  /* Left slice */
  if (threadIdx.x < radius_2) {
    int offset = blockDim.x * blockIdx.x - radius_2 + threadIdx.x;
    size_t global_x_l = offset >= 0 ? offset : -offset;
    shared[SIND(-radius_2 + threadIdx.x, threadIdx.y)] = input[IND(global_x_l, global_y)];
  }

  /* Right slice */
  if (threadIdx.x > blockDim.x - 1 - radius_2) {
    int index = blockDim.x - threadIdx.x;
    int offset = blockDim.x *(blockIdx.x + 1) + radius_2 - index;
    size_t global_x_r = offset < width ? offset : 2 * width - offset - 2;
    shared[SIND(radius_2 + threadIdx.x, threadIdx.y)] = input[IND(global_x_r, global_y)];
  }

  /* Upper slice */
  if (threadIdx.y < radius_2) {
    int offset = blockDim.y * blockIdx.y - radius_2 + threadIdx.y;
    size_t global_y_u = offset >= 0 ? offset : -offset;
    shared[SIND(threadIdx.x, -radius_2 + threadIdx.y)] = input[IND(global_x, global_y_u)];
  }

  /* Bottom slice */
  if (threadIdx.y > blockDim.y - 1 - radius_2) {
    int index = blockDim.y - threadIdx.y;
    int offset = blockDim.y *(blockIdx.y + 1) + radius_2 - index;
    size_t global_y_b = offset < height ? offset : 2 * height - offset - 2;
    shared[SIND(threadIdx.x, radius_2 + threadIdx.y)] = input[IND(global_x, global_y_b)];
  }


  
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //
  // -------------------------------------------------------- //
  __syncthreads();
  // -------------------------------------------------------- //
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

  if (global_id.x < width && global_id.y < height) {
    float buffer[49]; /* Max supported radius is 7, we have to store 7*7*7 values. */
      for (size_t iy = 0; iy < radius; ++iy) {
        for (size_t ix = 0; ix < radius; ++ix) {
          size_t lx = threadIdx.x - ix + radius_2;
          size_t ly = threadIdx.y - iy + radius_2;
          buffer[iy * radius + ix] = shared[SIND(lx, ly)];
        }
      }
    

    size_t length = radius * radius;
    sort(buffer, length);

    output[IND(global_id.x, global_id.y)] = buffer[length / 2];
  }
}