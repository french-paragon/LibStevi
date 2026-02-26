#ifndef COMBINATORIAL_H
#define COMBINATORIAL_H

#include <vector>
#include <cmath>

namespace StereoVision {

/*!
 * The combinatorial namespace contain utils for combinatorial indexing and stuff
 */
namespace Combinatorial {

/*!
 * \brief The ChooseInSetIndexer class allow to quickly index the possible choice of k elements in a set of N (permutation invariant)
 *
 * this class rely on a lot of undefined behavior for optimizations, pay attention to the docstring of each function!
 */
class ChooseInSetIndexer {
public:
	inline ChooseInSetIndexer(int n, int k) :
		_setSize(n),
		_nChoose(k)
	{

	}

	inline static int nChooseK(int n, int k) {
		int num = 1;
		int denum = 1;
		for (int i = 1; i <= k; i++) {
			num *= n - (k - i);
			denum *= i;
		}
		return num/denum;
	}

	inline int nChoices() const {
		return nChooseK(_setSize, _nChoose);
	}

	std::vector<int> idx2set(int idx) {
		std::vector<int> ret(_nChoose);
		int current = idx;
		int prev = -1;
		for (int i = 0; i < _nChoose; i++) {
			int min = prev+1;
			int max = _setSize-_nChoose+i;

			for (int s = min; s <= max; s++) {
				int n = _setSize-s-1;
				int k = _nChoose-i-1;
				int incr = nChooseK(n,k);
				ret[i] = s;
				if (incr > current) {
					break;
				}
				current -= incr;
			}
			prev = ret[i];
		}

		return ret;
	}

	/*!
	 * \brief set2idx get the index of a possible choice set
	 * \param set the choice of k elements, must be sorted and each element must be unique.
	 * \return an index between 0 and nSets()-1
	 *
	 * If the choice set is not sorted, not of size k or contain indices >= n, then the behavior is undefined.
	 */
	int set2idx(std::vector<int> const& choice) {
		int idx = 0;
		int prev = -1;
		int currentSetSize = _setSize;
		for (int i = 0; i < _nChoose; i++) {
			int delta = choice[i]-prev-1;
			for (int s = 0; s < delta; s++) {
				int n = currentSetSize-s-1;
				int k = _nChoose-i-1;
				idx += nChooseK(n,k);
			}
			currentSetSize -= delta+1;
			prev = choice[i];
		}
		return idx;
	}

protected:
	int _setSize;
	int _nChoose;
};

}

} // namespace StereoVision

#endif // COMBINATORIAL_H
