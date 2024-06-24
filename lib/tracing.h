#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ray.h"
#include "sphere.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "structs.h"
#include "camera.h"

cv::Mat renderScene(cv::cuda::GpuMat hdri, int img_w, int img_h, float& angle, sphere_t *scene, disk_t* disk, camera_t *cam); // BASELINE
cv::Mat renderScene(cv::cuda::GpuMat hdri,int img_w, int img_h, float& angle); // SHARED
cv::Mat renderSceneConst(cv::cuda::GpuMat hdri, int img_w, int img_h, float& angle); // CONSTANT

sphere_t* createSceneStruct(float angle, cudaStream_t stream);
void createSceneInConstant(float angle,cudaStream_t stream, camera_t* cam);

