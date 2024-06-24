# Blackhole-light-bending-simulation
A simulation of gravitational lensing due to blackhole space distortion done by exploiting GPU's versatility in C++ CUDA platform and OpenCV for the rendering part. 
Done in collaboration with @vinz321.

# 1 Introduction

## 1.1 Black Hole

The mysterious cosmic objects that are so dense that nothing can escape their gravity, neither light. 

A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole. The boundary of no escape is called the event horizon.

## 1.2 Gravitational Lensing

A gravitational lens is matter, such as a cluster of galaxies or a point particle, that bends light from a distant source as it travels toward an observer. The amount of gravitational lensing is described by Albert Einstein's general theory of relativity with much greater accuracy than Newtonian physics, which treats light as corpuscles travelling at the speed of light.

## 1.3 Ray-Marching

Is a class of rendering techniques similar to ray-casting that divides a single ray in many smaller rays of length d, that may be variable o not depending on the specific algorithm, and at each step it samples some function to determine the color to render.
The main advantage is that it allows to render objects according to their implicit function, this way it’s possible to render shapes for which it’s impossible or complex to calculate a ray-surface intersection.
It appeared to be the best-fit rendering algorithm for our purposes due to its discrete nature and its direct correlation with the light dynamics in space. In our specific case, photons behavior in space could be sufficiently approximated by ray-marching after a slightly customization needed to make rays correctly react to the gravitational pull of the blackhole.  

# 2. Algorithm

## 2.1 Our customization

As stated before, it was necessary to customize the ray-marching algorithm to obey to realtivistic laws of gravitational lensing. 

Just as ray-tracing, ray-marching simply casts as many rays as pixels we want the final render to have but instead of using continuous, straight lines, that doesn’t help us as we move in curved space, it make little steps toward a given direction and check for intesections with the world objects. In our specific case, within the context of a single sub-ray we not only had to check intersections with ordinary shapes but also if the ray entered the range of action of the blackhole’s gravitational pull and than deviate it by a certain angle. This angle is directly given by the following formula:

$$
\theta=\frac{2GM}{c^2r}
$$

This formula is a Newtonian approximation of light deflection’s angle given the gravitational constant, the mass, the speed of light and the distance from the blackhole. So, at every step, the sub-ray gets rotated by the resulted angle and then proceeds in that direction.

Although the result was satisfying enough in terms of graphical aspect, it required the use of trigonometric functions and a considerable overhead.

A much more efficient solution was a result of the following formula:

$$
\vec F(r)=-\frac{3}{2}h\frac{\hat r}{r^5}
$$

Where h is a constant and the value of that has a physical meaning not relevant to the final visual effect, thus we found the proper value by trial and error

```cpp
For each i in 0 -> ray.n_steps:
		Next_origin=ray.dir+ray.origin
		If next_origin interescts object O:
				Return O.color
Ray.dir=deflect_ray(next_origin, blackhole)
Ray.origin=next_origin
Return raycast(hdri, ray.dir)
```

# 3. GPU Optimization

## 3.1 Pseudo-Code

The most intuitive way of calculating the render is to cast one single ray per pixel and then start the ray-marching process. The main limitation to the algorithm is that each step n depends on the previous step n-1 forcing the algorithm to be sequential for each thread in the GPU.
```cpp
 Create scene
Load scene to device memory
Call render kernel per-pixel:
		Initialize ray
		For each i in 0 -> ray.n_steps:
				Next_origin=ray.dir+ray.origin
				If next_origin interescts object O:
						Return O.color
		Ray.dir=deflect_ray(next_origin, blackhole)
		Ray.origin=next_origin
		Return raycast(hdri, ray.dir)
``` 

Due to the nature of the problem it was not possible to avoid the for-loop, which is a serious disadvantage to the parallel calculation, it slows down considerably the calculations due to the waits caused by loop iterations and high branching factor that causes divergence in threads making the SIMD paradigm hard to execute.

### 3.2  Register usage

The compiler chooses whether to save variables in registers, local or shared memory. 
For the code we wrote, the compiler uses intensively the local memory instead of using registers.
On a GTX 3060, the compiler provides 58 registers per thread, limiting the number of blocks that can be scheduled in parallel.

Further limiting the number of registers would result in a better occupancy percentage that could hide the uncoalesced global accesses.

When calling device functions the compiler saves the content of the registers in local memory waiting for the context to return. On return it has to load back the content of the register resulting in an overhead, a solution would be to refactor the code in order to make less use of functions sacrificing the readability. 

## 3.3 Memory

For each kernel it is necessary to instantiate the information for a ray, which are: the current position and the direction,  and some helper variables to store intermediate values necessary for the final results, the colour which will store the colour of the first object hit by the ray, the unit vector from the ray sample to the black hole *r* and the distance from the black hole.

Every vector is made of 3 floats (12B), and a ray is composed by 2 vectors one for the sample position and one for the sample direction.

For this reason our project results in some uncoalesced accesses as reported by the profiler.                                                                                                                                                                                                                                                                                                                                        

An attempt to align data in global memory results in a more important overhead.
A cache line is 128 bytes wide, thus it can contain a total of 10 full vec3_t and 2 floats of the eleventh.
If we align the vec3_t to 16 bytes, a cache line would contain a total of 8 vec3_t resulting in even more accesses to the memory. For this reason we decided to keep the uncoalesced version of the vectors.

The same happens to shared memory causing uncoalesced accesses, in that case it happens because it was not possible to find a configuration to make thread access all banks needed, not even with a stride.

## 3.4 GPU Speed Of Light

We analysed the GPU speed of light calculated by the profiler in 3 different cases. With a base line of scene stored in the global memory, in a situation in which we stored the scene in constant memory and in a case in which rays were stored in shared memory.

In all 3 of the cases the number of registers per thread is 40.
With the code we used the 3 version are all comparable and very close to their theoretical limit for the actual arithmetic intensity.

## 3.5 Occupancy

With 40 registers the theoretical occupancy rises up to 100%, filling up the warp.

To achieve the theoretical occupancy of 100% we had to refactor the code in order to minimize the number of registers and we had to change the number of threads per block, from a maximum of 1024 (with a shape of 32x32) to 128 (with a shape of 16x8). 
## 4. Conclusions

Due to the nature of the application, it was difficult to optimize and tune all the possible configuration that CUDA leaves to the programmer: tweaking one component drastically worsen the others. However, we somehow found a satisfying equilibrium of all this configurations and in the end we reached a not-so-bad result, even graphically pleasing.
