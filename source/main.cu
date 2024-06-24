#include "cuda_profiler_api.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include "tracing.h"
#include <math.h>
#include "sky_engine.h"
using namespace cv::cuda;

int main() {
	float img_w = 512;
	float img_h = 256;
	float angle = 0;
	float time;
	vec3_t cam_pos= vec3_t{ 0,2*sinf(angle),-2*cosf(angle)};
	vec3_t cam_dir= vec3_t{ 0,-sinf(angle),cosf(angle)};
	Mat3f hdri_cpu = hdriread();
	size_t free, total;
	cudaEvent_t start, end;
	cudaStream_t mem_stream;
	GpuMat hdri;

	cudaStreamCreate(&mem_stream);
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	//BASELINE
	sphere_t* scene = createSceneStruct(0, mem_stream);

	cam_pos = vec3_t{ 0, 0, 0 };
	cam_dir = vec3_t{ 0, 0, 1 };
	
	cv::resize(hdri_cpu, hdri_cpu, Size(1024, 512));
	hdri.upload(hdri_cpu);
	
	for(int i=0;i<5;i++)
	//while(true)
	{
		if(i==4)
			cudaProfilerStart();
		cudaEventRecord(start);

		cam_pos = vec3_t{ 2*cosf(angle), 2* sinf(angle), -0.25f};
		cam_dir = norm(vec3_t{0,0,0} - cam_pos);
		camera_t cam = make_cam(cam_pos, cam_dir, vec3_t{ 0,0,1}, 60, (float)img_w / img_h);

		//BASELINE (REMOVE BASELINE COMMENT OUTSIDE LOOP)
		scene = createSceneStruct(angle, mem_stream);
		cv::Mat m = renderScene(hdri, img_w, img_h, angle, scene, (disk_t *)(scene + 3), &cam); 

		//CONSTANT
		/*createSceneInConstant(2 * angle, mem_stream, &cam);
		cv::Mat m = renderSceneConst(hdri, img_w, img_h, angle);*/

		//SHARED
		/*createSceneInConstant(2 * angle, mem_stream, &cam);
		cv::Mat m = renderScene(hdri, img_w, img_h, angle);*/


		cudaEventRecord(end);
		
		cv::imshow("Output", m);
		angle += 0.1f;
		
		cudaEventElapsedTime(&time, start, end);
		std::cout << "Time elapsed: " << time << std::endl;

		if ((cv::waitKey(1) & 0xFF) == 'q') {
			cudaProfilerStop();
			break;
		}
	}

	return 0;
}
