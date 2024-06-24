#include "camera.h"

__host__ __device__ camera_t make_cam(vec3_t lookfrom, vec3_t lookat, vec3_t vup, float vfov, float aspect) {
    vec3_t u, v, w;
    camera_t cam;
    float theta = vfov * PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    cam.origin = lookfrom;
    w = norm(lookfrom - lookat);
    u = norm(cross(vup, w));
    v = cross(w, u);
    cam.lower_left_corner = cam.origin - half_width * u - half_height * v - w;
    cam.horizontal = 2 * half_width * u;
    cam.vertical = 2 * half_height * v;

    return cam;
}