#include "ray.h"

//SHARED
__device__ vec3_t march(ray_t* r, cv::cuda::PtrStepSz<vec3_t> hdr, sphere_t* obj_ls, sphere_t* blackhole, int count, disk_t* disk) {
	//vec3_t next_orig;
	vec3_t color = { -1,-1,-1 };
	vec3_t r_g = blackhole->position - r->orig;
	float d = r_g * r_g;
	/*vec3_t t = cross(r->dir, norm(blackhole->position - r->orig));
	vec3_t k = norm(t);
	*/

	bool done = false;
	#pragma unroll 4
	for (int i = 0; i < N_STEPS; i++) {

		r->orig = r->orig + DELTA * r->dir;
		r_g = blackhole->position - r->orig;
		d = r_g * r_g;
		//t = cross(r->dir, norm(blackhole->position - r->orig));
		done|=is_inside(*blackhole, r_g, r->orig, color);
		done|=hit_disk(*disk, r, color);
		/*if (is_inside(*blackhole, r_g, r->orig, color)) {
			return color;
		}

		if (hit_disk(*disk, r, color)) {
			return color;
		}*/

		for (int j = 0; j < count; j++) {
			/*if (is_inside(obj_ls[j], r->orig, color)) {
				return color;
			}*/
			done|=is_inside(obj_ls[j], r->orig, color);
		}

		d = (3.0f / 2.0f) * 0.00005f / (d * d * d);
		r->dir = norm(r->dir + d * r_g);

		if (done) {
			break;
		}
	}
	if(color.x<0)
		color = hdr((((asinf((r->dir.y)) + (PI / 2)) / PI)) * 512, (1 - (atan2f(r->dir.z, r->dir.x) + PI) / (2 * PI)) * 1024);
	return color;
}

//BASELINE & CONSTANT
__device__ vec3_t march(ray_t& r, cv::cuda::PtrStepSz<vec3_t> hdr, sphere_t* obj_ls, sphere_t* blackhole, int count, disk_t* disk) {
	//vec3_t next_orig;
	vec3_t color;
	//vec3_t t = cross(r.dir, norm(blackhole->position - r.orig));
	vec3_t r_g = blackhole->position - r.orig;
	float d = r_g * r_g;
	//vec3_t k = norm(t);

	for (int i = 0; i < N_STEPS; i++) {

		r.orig = r.orig + DELTA * r.dir;
		r_g = blackhole->position - r.orig;
		d = r_g * r_g;

		//t = cross(r.dir, norm(blackhole->position - r.orig));

		if (is_inside(*blackhole, r_g, r.orig, color)) {
			return color;
		}

		if (hit_disk(*disk, r.orig, r.dir, color)) {
			return color;
		}

		for (int j = 0; j < count; j++) {
			if (is_inside(obj_ls[j], r.orig, color)) {
				return color;
			}
		}

		r.dir = norm(r.dir + (3.0f / 2.0f) * 0.00005f * r_g / (d * d * d));
		//r.dir = norm(rotate(r.dir, k, get_deflection(*blackhole, next_orig) * (t * t)));
		//r.orig = next_orig; 
	}
	color = hdr((((asinf((r.dir.y)) + (PI / 2)) / PI)) * 512, (1 - (atan2f(r.dir.z, r.dir.x) + PI) / (2 * PI)) * 1024);
	return color;
}
