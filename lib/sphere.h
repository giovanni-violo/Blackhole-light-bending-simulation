#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <iostream>

 __device__ bool is_inside(sphere_t &sphere, vec3_t point, vec3_t& col);
 __device__ bool is_inside(sphere_t& sphere, vec3_t r, vec3_t point, vec3_t& col);

__host__ __device__ float get_deflection(sphere_t &sphere, vec3_t point);