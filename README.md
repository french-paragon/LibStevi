# LibStevi
LibStevi is a collection of utilities for 3D computer vision, especially stereo vision, written in c++

## Dependencies

The IO module depends on the Cimg Library [https://cimg.eu/], a single header libary for image processing. The LibStevi IO module is a thin wrapper around Cimg and its capabilities will be defined by the the Cimg capabilities (which depends on the presence of certain software or libraries like imagemagick on the target system).

The tests and examples also require a Qt installation on the target system.

Last but not list, the library depends on the pnp library [https://github.com/midjji/pnp], which is provided as a git submodule.

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

The buildTests and buildExamples options are ON by default.

## Test data

The test data can be downloaded apart (to not clog the main repository) from [here](https://drive.google.com/file/d/1ybYTbgTyB7N1rCmJU0aXim2ElwUlOzkK/view?usp=sharing) and copied in /test (so that you get a folder /test/test_data/ in your source tree).

We took some test data from the Active-Passive SimStereo dataset [https://ieee-dataport.org/open-access/active-passive-simstereo]. This dataset is licensed under a Creative Commons Attribution license, if you reuse this data, you have to cite the original work:

	@data{gf1e-t452-22,
	doi = {10.21227/gf1e-t452},
	url = {https://dx.doi.org/10.21227/gf1e-t452},
	author = {Jospin, Laurent Valentin and Laga, Hamid and Boussaid, Farid and Bennamoun, Mohammed},
	publisher = {IEEE Dataport},
	title = {Active-Passive SimStereo},
	year = {2022}} 


