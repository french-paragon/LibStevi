#ifndef STEREO_VISION_GENERIC_RANSAC_H
#define STEREO_VISION_GENERIC_RANSAC_H

/*LibStevi, or the Stereo Vision Library, is a collection of utilities for 3D computer vision.

Copyright (C) 2025  Paragon<french.paragon@gmail.com>

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

#include <vector>
#include <random>
#include <mutex>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace StereoVision {
namespace Optimization {

/*!
 * \brief The GenericRansac class is an helper class to build generic ransac optimisator
 *
 * The classe takes two template arguments, the measure, which has to be copy-constructible, and the Model.
 *
 * The model can be built from a collection of measures, and then gives an adequacy metric for all measures.
 * The model has to be copy constructible, constructible from a vector of measures and default constructible.
 * It is recommanded that move semantics be supported.
 * The model also require a method, error, taking as input a measurement and returning a type implicitly castable to double.
 */
template<typename Measure, typename Model>
class GenericRansac {
public:
    GenericRansac(std::vector<Measure> const& measures, int minimumMeasures, double inlierThreshold) :
        _measures(measures),
        _minimumMeasures(minimumMeasures),
        _threshold(inlierThreshold),
        _nInliers(0),
        _currentInliers(measures.size()),
        _currentModel()
    {
        std::default_random_engine re;

        _re.seed(re());

    }

    GenericRansac(std::vector<Measure> && measures, int minimumMeasures, double inlierThreshold) :
        _measures(std::move(measures)),
        _minimumMeasures(minimumMeasures),
        _threshold(inlierThreshold),
        _nInliers(0),
        _currentInliers(measures.size()),
        _currentModel()
    {
        std::default_random_engine re;

        _re.seed(re());
    }

    inline int currentNInliers() const {
        return _nInliers;
    }

    inline Model const& currentModel() const {
        return _currentModel;
    }

    inline std::vector<bool> const& currentInliers() const {
        return _currentInliers;
    }

    /*!
     * \brief ransacIteration run one iterator of the ransac loop
     */
    inline void ransacIteration() {
        ransacIterationImpl(_re);
    }


    /*!
     * \brief ransacIterations perform a large number of ransac iterations
     * \param nIterations the number of iterations to perform
     *
     * The function is optimized to try a large number of iterations at once using OpenMP (if available).
     * Each thread will initialize its own random engine, which can be expensive.
     */
    inline void ransacIterations(int nIterations) {

        #if defined(_OPENMP)

        auto salt = _re();

        #pragma omp parallel
        {
            std::default_random_engine thread_re; //one random engine per thread
            thread_re.seed(salt + omp_get_thread_num()); //seed the generator in each thread with the thread id and the salt
            #pragma omp for
            for (int i = 0; i < nIterations; i++) {
                ransacIterationImpl(thread_re);
            }
        }

        #else

        for (int i = 0; i < nIterations; i++) {
            ransacIterationImpl(_re);
        }

        #endif

    }

protected:

    template <typename RandomEngine>
    inline void ransacIterationImpl(RandomEngine & re) {
        std::vector<int> selectedIdxs = select_samples(re);

        std::vector<Measure> selectedMeasures(_minimumMeasures);

        for (int i = 0; i < _minimumMeasures; i++) {
            selectedMeasures[i] = _measures[selectedIdxs[i]];
        }

        Model candidate(selectedMeasures);

        int nCandidateInlier = 0;
        std::vector<bool> candidateInliers(_measures.size());

        for (int i = 0; i < _measures.size(); i++) {
            Measure const& m = _measures[i]; //enfore model be const, for thread safety.
            double error = candidate.error(m);

            if (error < _threshold) {
                candidateInliers[i] = true;
                nCandidateInlier ++;
            } else {
                candidateInliers[i] = false;
            }
        }


        #if defined(_OPENMP)
        if (nCandidateInlier > _nInliers) { //most of the time the model will not be better, so check once before locking the mutex
                                            //_nInliers can onle grow, so no risk of missing something if nCandidateInlier < _nInliers
            _selected_access_locker.lock();
            if (nCandidateInlier > _nInliers) { //need to check again after locking the mutex, just to avoid race conditions
                _nInliers = nCandidateInlier;
                _currentInliers = candidateInliers;
                _currentModel = candidate;
            }
            _selected_access_locker.unlock();
        }
        #else
        if (nCandidateInlier > _nInliers) {
            _nInliers = nCandidateInlier;
            _currentInliers = candidateInliers;
            _currentModel = candidate;
        }
        #endif
    }

    template <typename RandomEngine>
    inline std::vector<int> select_samples(RandomEngine & re) const {
        size_t range = _measures.size()-1;

        std::vector<int> ret(_minimumMeasures);

        for (int i = 0; i < _minimumMeasures; i++) {
            std::uniform_int_distribution<int> dist(0,range);
            int idx = dist(re);
            for (int j = 0; j < i; j++) {
                idx += (ret[j] <= idx) ? 1 : 0;
            }
            ret[i] = idx;
            range--;
        }

        return ret;
    }

    std::vector<Measure> _measures;
    int _minimumMeasures;
    double _threshold;

    std::mutex _selected_access_locker;
    int _nInliers;
    std::vector<bool> _currentInliers;
    Model _currentModel;

    std::default_random_engine _re;

};

} // namespace Optimization
} // namespace StereoVision

#endif // GENERIC_RANSAC_H
