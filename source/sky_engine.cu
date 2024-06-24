#pragma once
#include "sky_engine.h"

 cv::Mat3f hdriread(){
	Mat hdr = imread("hdri/milkyway.jpg");
	hdr.convertTo(hdr, CV_32FC3, 1/255.0f);
	
	return hdr;
}