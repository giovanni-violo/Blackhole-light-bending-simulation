#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"

__host__ __device__ vec3_t operator+ (const vec3_t& x, const vec3_t& y) {
	float a = x.x + y.x;
	float b = x.y + y.y;
	float c = x.z + y.z;
	return vec3_t{ a,b,c };
}

__host__ __device__ float operator* (const vec3_t& x, const vec3_t& y) {
	float a = x.x * y.x;
	a = x.y * y.y + a;
	a = x.z * y.z + a;

	return a;
}

__host__ __device__ vec3_t operator* (const float& y, const vec3_t& x) {
	return vec3_t{ x.x * y,x.y * y, x.z * y };
}

__host__ __device__ vec3_t operator- (const vec3_t& x, const vec3_t& y) {
	float a = x.x - y.x;
	float b = x.y - y.y;
	float c = x.z - y.z;

	return vec3_t{ a,b,c };
}

__host__ __device__ vec3_t operator/ (const vec3_t& x, const float& y) {
	return vec3_t{ x.x / y,
		x.y / y,
		x.z / y };
}

__host__ __device__ float mul_add(const vec3_t& x, const vec3_t& y, float c) {
	float a = x.z * y.z + c;
	a = x.y * y.y + a;
	a = x.x * y.x + a;
	return a;
}

__host__ __device__ vec3_t norm(const vec3_t& v) {
	return v / sqrtf(v * v);
}

__host__ __device__ vec3_t cross(const vec3_t& x, const vec3_t& y) {
	float a=x.y* y.z - x.z * y.y;
	float b = -(x.x * y.z - x.z * y.x);
	float c = x.x * y.y - x.y * y.x;
	
	return vec3_t{ a,b,c };
	//return vec3_t{ x.y * y.z - x.z * y.y, -(x.x * y.z - x.z * y.x), x.x * y.y - x.y * y.x };
}

__host__ __device__ vec3_t rotate(const vec3_t& x, const vec3_t& k, float theta) {
	float cos = cosf(theta);
	float sin = sinf(theta);
	vec3_t a = cos * x;
	a = a+ sin * cross(k, x);
	float b = (k * x) * (1 - cos);
	vec3_t c = b * k;
	
	return a+c;
}

__host__ __device__ bool hit_disk(disk_t& disk, vec3_t& point, vec3_t& dir, vec3_t& color) {
	float dot = disk.normal * dir; //Dot proudct normal and ray direction
	vec3_t rel_pos = (point - disk.position); //relative position point to disk
	float d = rel_pos * disk.normal; //distance point - diskplane

	if (d * dot > 0 || d > DELTA || d < -DELTA) {
		return false;
	}

	vec3_t plane_pos = rel_pos - d * disk.normal;
	float r = plane_pos * plane_pos;

	if (r<disk.radius1* disk.radius1 || r>disk.radius2* disk.radius2) {
		return false;
	}
	color = disk.color;
	return true;
}

__host__ __device__ bool hit_disk(disk_t& disk, ray_t *ray, vec3_t& color) {
	float dot = disk.normal * ray->dir; //Dot proudct normal and ray direction
	vec3_t rel_pos = (ray->orig- disk.position); //relative position point to disk
	float d = rel_pos * disk.normal; //distance point - diskplane

	if (d * dot > 0 || d > DELTA|| d < - DELTA) {
		return false;
	}

	vec3_t plane_pos = rel_pos - d * disk.normal;
	float r = plane_pos * plane_pos;

	if (r<disk.radius1*disk.radius1 || r>disk.radius2 * disk.radius2) {
		return false;
	}
	color = disk.color;
	return true;
}