#pragma once

#include "tracing.h"
#include "sky_engine.h"
#include <iostream>

using namespace std;
using namespace cv;

__constant__ sphere_t scene_const[8];
__constant__ camera_t cam_const[1];
cv::cuda::GpuMat gpu_img;

//BASELINE
/*This section contains the baseline (no constant, no shared mem) version of the application*/
sphere_t* createSceneStruct(float angle, cudaStream_t stream) {
    int size = 2;

    sphere_t* scene;
    cudaMallocHost(&scene, sizeof(sphere_t) * (size + 1) + sizeof(disk_t));
    sphere_t* scene_gpu;
    cudaMalloc(&scene_gpu, sizeof(sphere_t) * (size + 1) + sizeof(disk_t));

    scene[0] = sphere_t{ { 0,0,0 }, 0.2f, { 0,0,0 } , 0.0025f };
    scene[1] = sphere_t{ vec3_t{ -.8f * cosf(angle) , .8f * sinf(angle), 0 }, 0.1f, { 1,1,.8f }, 0 };
    scene[2] = sphere_t{ vec3_t{ 0.95f,0,0 }, 0.05f, {.9f ,1, 1 }, 0 };

    *(disk_t*)(scene + 3) = disk_t{ {0,0,0}, 0.25f, 0.6f, {1,1,1} ,{0,0,1} };

    cudaMemcpyAsync(scene_gpu, scene, sizeof(sphere_t) * (size + 1) + sizeof(disk_t), cudaMemcpyHostToDevice);

    return scene_gpu;
}

__global__ void render_base(cv::cuda::PtrStepSz<vec3_t> img, cv::cuda::PtrStepSz<vec3_t> hdr, int max_x, int max_y, camera_t* cam_o, sphere_t* ls, int count, disk_t* disk_s) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //sphere s = sphere(*ls[0]);
    if ((i >= max_x) || (j >= max_y)) return;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    //ray r = ray(cam->origin, cam->lower_left_corner + (u * cam->horizontal) + (v * cam->vertical) - cam->origin, i, j, hdr);
    ray_t r1 = ray_t{ cam_o->origin, cam_o->lower_left_corner + (u * cam_o->horizontal) + (v * cam_o->vertical) - cam_o->origin };
    //vec3_t col = r.march(ls, blackhole, count, disk);
    vec3_t col = march(r1, hdr, ls + 1, ls, count, disk_s);
    img(j, i) = col;
}

cv::Mat renderScene(cv::cuda::GpuMat hdri, int img_w, int img_h, float& angle, sphere_t* scene, disk_t* disk, camera_t* cam) {
    cv::Mat3f img(img_h, img_w);
    if(gpu_img.empty())
        gpu_img.upload(img);

    dim3 grid_size(img_w / 16, img_h / 8);
    dim3 block_size(16, 8);
    camera_t* cam_gpu;

    cudaMalloc(&cam_gpu, sizeof(camera_t));
    cudaMemcpyAsync(cam_gpu, cam, sizeof(camera_t), cudaMemcpyHostToDevice);

    
    render_base << < grid_size, block_size >> > (gpu_img, hdri, img_w, img_h, cam_gpu, scene, 2, (disk_t*)disk);
 
    gpu_img.download(img);
    cudaFree(cam_gpu);
    return img;
}

//CONSTANT
/*This section contains the version of the application that uses constant memory to store all the scene objects*/

void createSceneInConstant(float angle, cudaStream_t stream, camera_t* cam) {
    int size = 3;

    sphere_t* scene;
    cudaMallocHost(&scene, sizeof(sphere_t) * size + sizeof(disk_t));

    scene[0] = sphere_t{ { 0,0,0 }, 0.2f, { 0,0,0 } , 0.0025f };
    scene[1] = sphere_t{ vec3_t{ -.8f * cosf(angle) , .8f * sinf(angle), 0 }, 0.1f, { 1,1,.8f }, 0 };
    scene[2] = sphere_t{ vec3_t{ 0.95f,0,0 }, 0.05f, {.9f ,1, 1 }, 0 };
    *(disk_t*)(scene + size) = disk_t{ {0,0,0}, 0.25f, 0.6f, {1,1,1} ,{0,0,1} };

    cudaMemcpyToSymbolAsync(scene_const, scene, sizeof(sphere_t) * size + sizeof(disk_t), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(cam_const, cam, sizeof(camera_t), 0, cudaMemcpyHostToDevice, stream);
}

__global__ void render_constant(cv::cuda::PtrStepSz<vec3_t> img, cv::cuda::PtrStepSz<vec3_t> hdr, int max_x, int max_y, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //sphere s = sphere(*ls[0]);
    if ((i >= max_x) || (j >= max_y)) return;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    //ray r = ray(cam->origin, cam->lower_left_corner + (u * cam->horizontal) + (v * cam->vertical) - cam->origin, i, j, hdr);
    ray_t r1 = ray_t{ cam_const->origin, cam_const->lower_left_corner + (u * cam_const->horizontal) + (v * cam_const->vertical) - cam_const->origin };

    //vec3_t col = r.march(ls, blackhole, count, disk);
    vec3_t col = march(r1, hdr, scene_const + 1, scene_const, count, (disk_t*)(scene_const + 1 + count));

    //__syncthreads();
    img(j, i) = col;
}

cv::Mat renderSceneConst(cv::cuda::GpuMat hdri, int img_w, int img_h, float& angle) {
    cv::Mat3f img(img_h, img_w);

    if (gpu_img.empty()) gpu_img.upload(img);

    dim3 grid_size(img_w /2,  img_h / 64);
    dim3 block_size(2, 64);

    render_constant <<< grid_size, block_size >> > (gpu_img, hdri, img_w, img_h, 2);
   
    gpu_img.download(img);
    
    return img;
}

//SHARED
/*This section contains a version of the application that uses shraed memory to store all the rays and constant memory to store all the scene objects*/
__global__ void render_shared(cv::cuda::PtrStepSz<vec3_t> img, cv::cuda::PtrStepSz<vec3_t> hdr, int max_x, int max_y, int count) {

    __shared__ ray_t rays[128];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int rid = threadIdx.x + threadIdx.y * blockDim.x;

    if ((i >= max_x) || (j >= max_y)) return;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    rays[rid] = ray_t{ cam_const->origin, cam_const->lower_left_corner + (u * cam_const->horizontal) + (v * cam_const->vertical) - cam_const->origin };

    //vec3_t col = 
    img(j, i) = march(rays + rid, hdr, scene_const + 1, scene_const, count, (disk_t*)(scene_const + count + 1));
}

cv::Mat renderScene(cv::cuda::GpuMat hdri, int img_w, int img_h, float& angle) {
    cv::Mat3f img(img_h, img_w);

    if (gpu_img.empty()) gpu_img.upload(img);

    dim3 grid_size(img_w / 16, img_h / 8);
    dim3 block_size(16, 8);

    render_shared << < grid_size, block_size >> > (gpu_img, hdri, img_w, img_h, 2);

    gpu_img.download(img);

    return img;
}
