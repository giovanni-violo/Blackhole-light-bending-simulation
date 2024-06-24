#pragma once

#include "structs.h"
#include "ray.h"

__host__ __device__ camera_t make_cam(vec3_t lookfrom, vec3_t lookat, vec3_t vup, float vfov, float aspect);