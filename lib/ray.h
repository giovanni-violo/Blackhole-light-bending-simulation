#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include "sphere.h"
#include "sky_engine.h"

__device__ vec3_t march(ray_t* r, cv::cuda::PtrStepSz<vec3_t> hdr, sphere_t* obj_ls, sphere_t* blackhole, int count, disk_t* disk);
__device__ vec3_t march(ray_t& r, cv::cuda::PtrStepSz<vec3_t> hdr, sphere_t* obj_ls, sphere_t* blackhole, int count, disk_t* disk);