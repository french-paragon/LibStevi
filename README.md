# LibStevi
LibStevi is a collection of utilities for 3D computer vision, especially stereo vision, written in c++

## Dependencies

The IO module depends on the Cimg Library [https://cimg.eu/], a single header libary for image processing. The LibStevi IO module is a thin wrapper around Cimg and its capabilities will be defined by the the Cimg capabilities (which depends on the presence of certain software or libraries like imagemagick on the target system).

The tests and examples also require a Qt installation on the target system.

## Get the code

You can get the code using git, pay attention to init and sync the submodules:

	mkdir src
	cd src
	git clone https://github.com/french-paragon/LibStevi
	cd LibStevi
	git submodule init --recursive
	git submodule update --recursive

## Build

The building process is straighforward with cmake:

	mkdir build
	cd ./build
	cmake ../path/to/src/LibStevi -DbuildTests=ON/OFF -DbuildExamples=ON/OFF
	make

The buildTests and buildExamples options are OFF by default.

## Use via FetchContent

To use the library in cmake directly using FetchContent write:

```
include(FetchContent) #if you have not done so before
FetchContent_Declare(
  StereoVision
  GIT_REPOSITORY https://github.com/french-paragon/LibStevi.git
  GIT_TAG [select your tag here]
  FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(StereoVision)
```

in your CMake Project. 

## Linking the library

You can link against the library by using in your CMakeLists.txt:

```
target_link_libraries([your_cmake_target] StereoVision::stevi)
```

## Test data

The test data can be downloaded apart (to not clog the main repository) from [here](https://drive.google.com/file/d/1ybYTbgTyB7N1rCmJU0aXim2ElwUlOzkK/view?usp=sharing) and copied in /test (so that you get a folder /test/test_data/ in your source tree).

We took some test data from the Active-Passive SimStereo dataset [https://ieee-dataport.org/open-access/active-passive-simstereo]. This dataset is licensed under a Creative Commons Attribution license, if you reuse this data, you have to cite the original work:

```
@inproceedings{NEURIPS2022_bc3a68a2,
author = {Jospin, Laurent and Antony, Allen and Xu, Lian and Laga, Hamid and Boussaid, Farid and Bennamoun, Mohammed},
booktitle = {Advances in Neural Information Processing Systems},
editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
pages = {29235--29247},
publisher = {Curran Associates, Inc.},
title = {Active-Passive SimStereo - Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods},
url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Paper-Datasets_and_Benchmarks.pdf},
volume = {35},
year = {2022}
}
```


