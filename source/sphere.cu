#include "sphere.h"

 __device__ bool is_inside(sphere_t& sphere, vec3_t point, vec3_t &col){
	vec3_t a;
	float sqr_rad = sphere.radius * sphere.radius;
	a = sphere.position - point;
	if (mul_add(a,a, -sqr_rad)>0)
		return false;

	col = sphere.color;
	return true;
}

 __device__ bool is_inside(sphere_t& sphere, vec3_t r, vec3_t point, vec3_t& col) {
	 if (r*r > sphere.radius * sphere.radius)
		 return false;

	 col = sphere.color;
	 return true;
 }

__host__ __device__ float get_deflection(sphere_t& sphere, vec3_t point) {
	vec3_t dist_vec = point - sphere.position;
	float dist = sqrt(dist_vec * dist_vec);

	return (GRAV_LIGHT_CONST * sphere.mass) / dist;
}