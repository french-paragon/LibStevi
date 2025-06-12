# Intrisic image decomposition example

Intrisic image decomposition is the process of estimating the reflectance (or albedo) and shading (or ligthing) from a single image.
This tool uses the Intrisic image decomposition methods implemented in libStevi, which are derived from the method proposed in:

Zhao, Qi, et al. "A closed-form solution to retinex with nonlocal texture constraints." IEEE transactions on pattern analysis and machine intelligence 34.7 (2012): 1437-1444.

To call the tool, use the following syntax:

```
./intrisic_img_decomposition path_to_image_file [path_to_output_directory] [--outputfiles/-o]
```

Arguments:

path_to_image_file: path to an image file (jpg, png, tiff, bmp, ... or libStevi intermediate format, .stevimg).
path_to_output_directory: path to the output directory for the output images (if not set, the images will be saved in the same directory as the input image.

Options:

--outputfiles (-o) : directly output files to disk, as .stevimg files, as they are probably meant to be consumed by another tool. If the application is compiled with GUI support, .jpg previews of the decomposition will also be exported.

