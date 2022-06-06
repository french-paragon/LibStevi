#ifndef STEREOVISION_RANDOMCACHE_H
#define STEREOVISION_RANDOMCACHE_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2022  Paragon<french.paragon@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <memory>
#include <functional>

namespace StereoVision {

namespace Random {


template<typename T>
/*!
 * \brief The NumbersCache class cache a serie of generated (random) numbers and then act as a random number generator.
 *
 * This class is meant to spead up pseudo random number generation for real time vision algorithms.
 * Note that operator(), which is meant to generate a "random" number is not thread safe or even reentrant.
 * If you want to use a NumbersCache in multiple threads, you first have to create a copy in each thread using the copy constructor.
 * The copy constructor keep a reference to the previously cached numbers but duplicate the current index of the cache.
 * To make sur eto generate the numbers at a different positions in the cache for each thread, you need to use the seed method with a different value on each copy.
 */
class NumbersCache {

public:

	NumbersCache () :
		_index(0),
		_size(1),
		_elements(new T [1])
	{

	}

	NumbersCache (size_t n, std::function<T()> const& generator) :
		_index(0),
		_size(n),
		_elements(new T [n])
	{
		for (size_t i = 0; i < n; i++) {
			_elements[i] = generator();
		}
	}

	NumbersCache (NumbersCache<T> const& other) :
		_index(other._index),
		_size(other.size()),
		_elements(other._elements)
	{

	}

	NumbersCache<T>& operator= (NumbersCache<T> const& other) {
		_index = other._index;
		_size = other._size;
		_elements = other._elements;

		return *this;
	}

	inline size_t size() const {
		return _size;
	}

	inline void seed(int idx) {
		_index += idx;
		size_t* reinterpreted_data = reinterpret_cast<size_t*>(_elements.get());
		int nSize = (_size*sizeof (T)) / sizeof (size_t);

		for (int i = 0; i < 8; i++) {
			_index += reinterpreted_data[_index % nSize];
		}

		_index %= _size;
	}

	inline T operator() () const {
		_index++;
		_index %= _size;
		return _elements[_index];
	}

private:

	mutable size_t _index;
	size_t _size;
	std::shared_ptr<T[]> _elements;

};

} // namespace Random

} // namespace StereoVision

#endif // STEREOVISION_RANDOMCACHE_H
