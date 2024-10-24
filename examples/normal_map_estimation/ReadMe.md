# Normal map estimation example

This example shows how to use the shape from shading functions in libStevi to estimate the normal map of a single image from its intrinsic decomposition.

The model is based on the lambertian surface assumption, assumes that the vertical angle of the incident light is 45°, by default assumes the light is coming from the top of the image (albeit it can estimate the direction when the -d swith is enabled, which it does by assuming a convex surface) and assumes the reconstructed surface has no linear trend (i.e., when fitting a plane to the reconstructed surface, this plane has constant z value).

To call the tool, use the following syntax:

```
./normal_map_estimation path_to_shading_file path_to_reflectance [path_to_area_of_interest_image]_file [-d] [-i]
```

Arguments:

path_to_shading_file: path to a .stevimg file with the estimated shading.
path_to_reflectance: path to a .stevimg file with the estimated reflectance.
path_to_area_of_interest_image: path to the region of interest. The software will enforce the non-linear trend constraint in that region only, and will plot the reconstructed surface in that region only. 

Options:

-d : estimate the light direction from the data, instead of the hardcoded top direction
-i : invert the convexity of the reconstructed surface. By default, to estimate the incoming light direction, the reconstructed surface is assumed to be mostly convex. This switch invert this hypothesis, which has the effect of inverting the planar direction of the incoming light.



